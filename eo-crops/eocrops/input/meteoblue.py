import aiohttp
import asyncio
from io import StringIO

import pandas as pd
import dateutil
import datetime as dt

import numpy as np
import datetime


class WeatherDownload:
    def __init__(self, api_key, queryBackbone, ids, coordinates, years, loop =  asyncio.new_event_loop()):
        '''

        Parameters
        ----------
        api_key (str) : API key from meteoblue
        queryBackbone (dict) : query backbone from meteoblue
        ids (list) : list of ids from each location request
        coordinates (list of tuples) : list of each location coordinate
        years (list) : years of extraction w.r.t the ids
        '''
        self.api_key = api_key
        self.queryBackbone = queryBackbone
        self.loop = loop
        self.ids = ids
        self.coordinates = coordinates
        self.years = years
        self.url_query = "http://my.meteoblue.com/dataset/query"
        self.url_queue = "http://queueresults.meteoblue.com/"

    async def _get_jobIDs_from_query(self, query, time_interval =  ('01-01', '12-31')):

        async def _make_ids(ids, coordinates, dates):
            for i, (id, coord, date) in enumerate(zip(ids, coordinates, dates)):
                yield i, id, coord, date

        jobIDs = []

        async for i, id, coord, date in _make_ids(self.ids, self.coordinates, self.years):
            await asyncio.sleep(0.5) #query spaced by 05 seconds => 2 queries max per queueTime (limit = 5)
            start_time, end_time = (str(date) + "-" + time_interval[0], str(date) + "-" + time_interval[1])

            self.queryBackbone["geometry"]["geometries"] = \
                [dict(type='MultiPoint', coordinates=[coord], locationNames=[id])]
            self.queryBackbone["timeIntervals"] = [start_time + 'T+00:00' + '/' + end_time + 'T+00:00']
            self.queryBackbone["queries"] = query

            async with aiohttp.ClientSession() as session:
                # prepare the coroutines that post
                async with session.post(self.url_query,
                                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                                        params={"apikey": self.api_key},
                                        json=self.queryBackbone
                                        ) as response:
                    data = await response.json()
                    print(data)
                await session.close()
            jobIDs.append(data['id'])
        # now execute them all at once
        return jobIDs


    async def _get_request_from_jobID(self, jobID, sleep = 1, limit = 5):

        await asyncio.sleep(sleep)
        #limit amount of simultaneously opened connections you can pass limit parameter to connector
        conn = aiohttp.TCPConnector(limit=limit, ttl_dns_cache=300)
        session = aiohttp.ClientSession(connector=conn) #ClientSession is the heart and the main entry point for all client API operations.
        #session contains a cookie storage and connection pool, thus cookies and connections are shared between HTTP requests sent by the same session.

        async with session.get(self.url_queue + jobID) as response:
            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])
            urlData = await response.text()
            print(response)
            await session.close()
        df = pd.read_csv(StringIO(urlData), sep=",", header=None)
        df['jobID'] = jobID
        return df

    @staticmethod
    async def _gather_with_concurrency(n, *tasks):
        semaphore = asyncio.Semaphore(n)
        async def sem_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(sem_task(task) for task in tasks))

    def execute(self, query, time_interval = ('01-01', '12-31'), conc_req = 5):
        try :
            jobIDs = self.loop.run_until_complete(self._get_jobIDs_from_query(query, time_interval))

            dfs = self.loop.run_until_complete(self._gather_with_concurrency(conc_req,
                                                                             *[self._get_request_from_jobID(jobID,
                                                                                                            i/100)
                                                                              for i, jobID in enumerate(jobIDs)]))
        finally:
            print('close')
            self.loop.close()

        return pd.concat(dfs, axis=0)

#############################################################################################

class WeatherPostprocess:
    def __init__(self,
                 input_file,
                 id_column,
                 year_column,
                 resample_range=('-01-01', '-12-31', 1),
                 planting_date_column = None, havest_date_column = None):
        '''
        Format output file from meteoblue API into a pd.DataFrame

        :param input_file (pd.DataFrame) : input file with fields in-situ data
        :param id_column (str): Name of the column that contains ids of the fields to merge with CEHub data
        :param planting_date_column (str): Name of the column with planting date in doy format
        :param havest_date_column (str): Name of the column with harvest date in doy format
        :param year_column (str) : Name of the column with the yearly season associated to each field
        :param resample_range (tuple): Query period (interval of date) and number of days to aggregate over period (e.g. 8 days) instead of having daily data
        '''

        self.id_column = id_column
        self.planting_date_column = planting_date_column
        self.havest_date_column = havest_date_column
        self.year_column = year_column
        self.resample_range = resample_range
        self.input_file = input_file[[self.id_column, self.year_column]].drop_duplicates()

        if self.planting_date_column is not None and self.havest_date_column is not None:
            self.input_file = self.input_file.rename(
                columns={self.planting_date_column: 'planting_date'})
            self._apply_convert_doy('planting_date')


    def _get_descriptive_period(self, df, stat='mean'):
        '''
        Compute descriptive statistics given period
        '''
        dict_stats = dict(mean=np.nanmean, max=np.nanmax, min=np.nanmin)

        df['value'] = df['value'].astype('float32')
        df_agg = df[['variable', 'period', 'location', 'value', self.year_column]]. \
            groupby(['variable', 'period', 'location', self.year_column]).agg(dict_stats[stat])
        df_agg.reset_index(inplace=True)
        df_agg = df_agg.rename(columns={'value': stat + '_value',
                                        'location': self.id_column})
        return df_agg

    def _get_cumulated_period(self, df):
        '''
        Compute the cumulative sum given period.
        '''

        df_cum = pd.DataFrame()
        for var in df.variable.unique():
            df_subset = df[df.variable == var]
            df_agg = df_subset[['location', 'period', 'variable', 'value', self.year_column]]. \
                groupby(['location', self.year_column, 'variable', 'period']).sum()
            df_agg = df_agg.groupby(level=0).cumsum().reset_index()
            df_agg = df_agg.rename(columns={'value': 'sum_value'})
            df_cum = df_cum.append(df_agg)

        return df_cum

    def _get_resampled_periods(self, year = '2021'):
        '''
        Get the resampled periods from the resample range
        '''
        resample_range_ = (str(year) + self.resample_range[0],
                           str(year) + self.resample_range[1],
                           self.resample_range[2])

        start_date = dateutil.parser.parse(resample_range_[0])
        end_date = dateutil.parser.parse(resample_range_[1])
        step = dt.timedelta(days=resample_range_[2])

        days = [start_date]
        while days[-1] + step < end_date:
            days.append(days[-1] + step)
        return days


    def _format_periods(self, periods):
        df_resampled = pd.melt(periods, id_vars='period'). \
            rename(columns={'value': 'timestamp', 'variable': self.year_column})

        # Left join periods to the original dataframe
        df_resampled['timestamp'] = [str(k) for k in df_resampled['timestamp'].values]
        df_resampled['timestamp'] = [np.datetime64(str(year) + '-' + '-'.join(k.split('-')[1:]))
                                     for year, k in zip(df_resampled[self.year_column], df_resampled['timestamp'])]
        return df_resampled

    def _get_periods(self, df_cehub_):
        '''
        Assign the periods to the file obtained through CEHub
        '''

        def _get_year(x): return x[:4]
        def _convert_date(x): return dateutil.parser.parse(x[:-5])

        df_cehub = df_cehub_.copy()

        # Assign period ids w.r.t the date from the dataframe
        df_cehub['timestamp'] = [str(k) for k in df_cehub['timestamp']]

        #Assign dates to a single year to retrieve periods
        df_cehub[self.year_column] = df_cehub['timestamp'].apply(lambda x: _get_year(x))
        df_cehub['timestamp'] = df_cehub['timestamp'].apply(lambda x: _convert_date(x))

        dict_year = {}
        for year in df_cehub[self.year_column].drop_duplicates().values:
            dict_year[year] = self._get_resampled_periods()

        periods = pd.DataFrame(dict_year)

        periods = periods.reset_index().rename(columns={'index': 'period'})
        df_resampled = self._format_periods(periods)

        df = pd.merge(df_resampled, df_cehub,
                      on=['timestamp', self.year_column],
                      how='right')
        #Interpolate over new periods
        fill_nas = df[['period', 'location']].groupby('location').apply(
            lambda group: group.interpolate(method='pad', limit=self.resample_range[-1]))

        df['period'] = fill_nas['period']

        return df, df[['period', 'timestamp']].drop_duplicates()


    def _apply_convert_doy(self, feature):
        '''
        Convert dates from CEhub format into day of the year
        '''
        def _convert_doy_to_date(doy, year):
            date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(doy - 1)
            return np.datetime64(date)

        self.input_file[feature] = [_convert_doy_to_date(doy, year)
                                    for doy, year in zip(self.input_file[feature],
                                                         self.input_file[self.year_column])]

    def _add_growing_stage(self, periods_df, feature='planting_date'):
        '''
        Retrive the date from weather data associated with a given growing stage (doy format) from the input file
        The objective is to not take into account observations before sowing date of after harvest date in the statistics
        '''

        return pd.merge(periods_df,
                        self.input_file[[feature, self.id_column]].copy(),
                        left_on='timestamp',
                        right_on=feature,
                        how='right'). \
            rename(columns={'period': 'period_' + feature}).drop(['timestamp'], axis=1)

    def _init_df(self, df):
        '''
        Initialize weather dataframe into periods to do the period calculations
        '''
        df = df[~df.variable.isin(['variable'])]
        df = df.drop_duplicates(subset=['location', 'timestamp', 'variable'])

        df = df[df.location.isin(self.input_file[self.id_column].unique())]
        df, periods_df = self._get_periods(df_cehub_=df)
        df['value'] = df['value'].astype('float32')

        if self.planting_date_column is not None:
            periods_sowing = self._add_growing_stage(periods_df, feature='planting_date')
            df = pd.merge(df[['period', 'timestamp', 'location', 'variable', 'value']],
                          periods_sowing,
                          left_on='location',
                          right_on= self.id_column,
                          how='left')

            # Observations before planting date are assigned to np.nan
            df.loc[df.timestamp < df.planting_date, ['value']] = np.nan

        return df

    def _prepare_output_file(self, df_stats, stat='mean'):
        '''
        Prepare output dataframe with associated statistics over the periods.
        The output will have the name of the feature and its corresponding period (tuple)
        '''
        df_pivot = pd.pivot_table(df_stats,
                                  values=[stat + '_value'],
                                  index=[self.id_column, self.year_column],
                                  columns=['variable', 'period'], dropna=False)

        df_pivot.reset_index(inplace=True)
        df_pivot.columns = ['-'.join([str(x) for x in col]).strip() for col in df_pivot.columns.values]
        df_pivot = df_pivot.rename(
            columns={
                self.id_column + '--': self.id_column,
                self.year_column + '--': self.year_column}
        )
        df_pivot = df_pivot.sort_values(by=[self.id_column, self.year_column]).reset_index(drop=True)
        return df_pivot


    def _get_temperature_difference(self, min_weather, max_weather):
        '''
        Compute difference between minimum and maximum temperature observed for each period
        '''
        diff_weather = min_weather.copy()

        tempMax = max_weather.loc[
            max_weather.variable.isin(['Temperature']),
            ['period', 'timestamp', self.id_column, 'value']].rename(
            columns={'value': 'value_max'})

        diff_weather = pd.merge(diff_weather,
                                tempMax,
                                on=['period', 'timestamp', self.id_column],
                                how='left')

        diff_weather['value'] = diff_weather['value_max'] - diff_weather['value']
        diff_weather['variable'] = 'Temperature difference'

        return diff_weather


    def execute(self, df_weather, stat='mean', return_pivot = False):
        '''
        Execute the workflow to get the dataframe aggregated into periods from CEHub data
        :param df_weather (pd.DataFrame) : cehub dataframe with stat as daily descriptive statistics
        :return:
            pd.DataFrame with mean, min, max, sum aggregated into periods defined w.r.t the resample_range
        '''

        if stat not in ['mean', 'min', 'max', 'sum', 'cumsum']:
            raise ValueError("Descriptive statistic must be 'mean', 'min', 'max', 'sum' or 'cumsum'")

        init_weather = self._init_df(df=df_weather.copy())

        if stat != 'cumsum':
            df_stats = self._get_descriptive_period(df = init_weather, stat = stat)
        else:
            df_stats = self._get_cumulated_period(df = init_weather)

        df_stats = df_stats.sort_values(by=[self.id_column, self.year_column, "variable"])

        if return_pivot:
            output = self._prepare_output_file(df_stats=df_stats, stat=stat)
            output.columns = [''.join(k.split('value-')) for k in output.columns]
            output.columns = [tuple(k.split('-')) if k != self.id_column else k for k in output.columns]
            output.columns = [(k[0], float(k[1])) if k != self.id_column else k for k in output.columns]

        return output
