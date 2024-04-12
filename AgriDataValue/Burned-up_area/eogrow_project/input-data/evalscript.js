//VERSION=3

function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "dataMask"],
      units: "DN"
    }],
    output: [
        {
            id: "BANDS",
            bands: 12,
            sampleType: SampleType.UINT16
        },
        {
            id: "dataMask",
            bands: 1,
            sampleType: SampleType.UINT8
        }
    ]
  };
}

function evaluatePixel(sample) {
  results = {
      "BANDS": [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12],
      "dataMask": [sample.dataMask]
  }
  return results;
}
