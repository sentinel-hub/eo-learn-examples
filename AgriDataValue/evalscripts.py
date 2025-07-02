evalscript_baresoil = """
//VERSION=3

//Author: Domagoj Korais

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B05", "B07", "B08", "B11"],
            units: "reflectance"
        }],
        output: {
            id:"baresoil",
            bands: 1,
            sampleType: "AUTO"
        }
    }
}

function evaluatePixel(sample) {
    var NBSI = ((sample.B11 + sample.B04)-(sample.B08 + sample.B02))/((sample.B11 + sample.B04)+(sample.B08 + sample.B02))
    var NDVI = index(sample.B08, sample.B04);
    var NDVI_RE3 = index(sample.B08, sample.B07);
    var CL_RE = sample.B07 / sample.B05 - 1;

    var is_bare_soil = NDVI <= -0.1 ? false : predict(NBSI, NDVI, NDVI_RE3, CL_RE) > 0.5;
    return [is_bare_soil]

}


var DecisionTreeClassifier = function() {

    var findMax = function(nums) {
        var index = 0;
        for (var i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    this.predict = function(features) {
        var classes = new Array(2);

        if (features[1] <= 0.2880808413028717) {
            if (features[2] <= -0.001884871511720121) {
                if (features[2] <= -0.01514277933165431) {
                    if (features[0] <= -0.05856157839298248) {
                        if (features[3] <= 0.4919503927230835) {
                            if (features[1] <= 0.22891760617494583) {
                                classes[0] = 109;
                                classes[1] = 75;
                            } else {
                                classes[0] = 77;
                                classes[1] = 5;
                            }
                        } else {
                            if (features[3] <= 0.6081486344337463) {
                                classes[0] = 47;
                                classes[1] = 128;
                            } else {
                                classes[0] = 22;
                                classes[1] = 311;
                            }
                        }
                    } else {
                        if (features[1] <= 0.23794686794281006) {
                            if (features[3] <= 0.24695706367492676) {
                                classes[0] = 51;
                                classes[1] = 132;
                            } else {
                                classes[0] = 352;
                                classes[1] = 3683;
                            }
                        } else {
                            if (features[3] <= 0.4766134023666382) {
                                classes[0] = 91;
                                classes[1] = 57;
                            } else {
                                classes[0] = 209;
                                classes[1] = 1278;
                            }
                        }
                    }
                } else {
                    if (features[0] <= -0.01485772943124175) {
                        if (features[3] <= 0.46598660945892334) {
                            if (features[1] <= 0.23513969033956528) {
                                classes[0] = 80;
                                classes[1] = 30;
                            } else {
                                classes[0] = 83;
                                classes[1] = 4;
                            }
                        } else {
                            classes[0] = 38;
                            classes[1] = 42;
                        }
                    } else {
                        if (features[1] <= 0.24381835758686066) {
                            if (features[0] <= 0.017081347294151783) {
                                classes[0] = 37;
                                classes[1] = 60;
                            } else {
                                classes[0] = 72;
                                classes[1] = 437;
                            }
                        } else {
                            if (features[3] <= 0.4962599277496338) {
                                classes[0] = 84;
                                classes[1] = 43;
                            } else {
                                classes[0] = 23;
                                classes[1] = 66;
                            }
                        }
                    }
                }
            } else {
                if (features[2] <= 0.012518306728452444) {
                    if (features[0] <= 0.011857263278216124) {
                        if (features[3] <= 0.44026511907577515) {
                            if (features[0] <= -0.025940910913050175) {
                                classes[0] = 155;
                                classes[1] = 2;
                            } else {
                                classes[0] = 122;
                                classes[1] = 13;
                            }
                        } else {
                            classes[0] = 58;
                            classes[1] = 38;
                        }
                    } else {
                        if (features[1] <= 0.21599827706813812) {
                            classes[0] = 44;
                            classes[1] = 86;
                        } else {
                            if (features[3] <= 0.4378824234008789) {
                                classes[0] = 145;
                                classes[1] = 37;
                            } else {
                                classes[0] = 57;
                                classes[1] = 55;
                            }
                        }
                    }
                } else {
                    if (features[3] <= 0.4603644013404846) {
                        if (features[1] <= 0.21943768113851547) {
                            if (features[0] <= 0.021366839297115803) {
                                classes[0] = 131;
                                classes[1] = 4;
                            } else {
                                classes[0] = 58;
                                classes[1] = 25;
                            }
                        } else {
                            if (features[3] <= 0.42150408029556274) {
                                classes[0] = 982;
                                classes[1] = 19;
                            } else {
                                classes[0] = 237;
                                classes[1] = 14;
                            }
                        }
                    } else {
                        classes[0] = 84;
                        classes[1] = 26;
                    }
                }
            }
        } else {
            if (features[2] <= -0.047297170385718346) {
                if (features[1] <= 0.40251147747039795) {
                    if (features[3] <= 0.6912856698036194) {
                        classes[0] = 114;
                        classes[1] = 25;
                    } else {
                        if (features[1] <= 0.3502514660358429) {
                            if (features[3] <= 0.7766227126121521) {
                                classes[0] = 29;
                                classes[1] = 74;
                            } else {
                                classes[0] = 64;
                                classes[1] = 516;
                            }
                        } else {
                            if (features[0] <= -0.027021611109375954) {
                                classes[0] = 84;
                                classes[1] = 236;
                            } else {
                                classes[0] = 72;
                                classes[1] = 30;
                            }
                        }
                    }
                } else {
                    if (features[1] <= 0.4671569764614105) {
                        if (features[0] <= -0.05327927693724632) {
                            if (features[2] <= -0.0706191249191761) {
                                classes[0] = 43;
                                classes[1] = 41;
                            } else {
                                classes[0] = 99;
                                classes[1] = 40;
                            }
                        } else {
                            if (features[1] <= 0.4272315502166748) {
                                classes[0] = 67;
                                classes[1] = 18;
                            } else {
                                classes[0] = 121;
                                classes[1] = 9;
                            }
                        }
                    } else {
                        if (features[3] <= 1.3283718824386597) {
                            classes[0] = 137;
                            classes[1] = 16;
                        } else {
                            if (features[2] <= -0.08075670152902603) {
                                classes[0] = 75;
                                classes[1] = 5;
                            } else {
                                classes[0] = 453;
                                classes[1] = 4;
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 0.3474765121936798) {
                    if (features[3] <= 0.5934110283851624) {
                        if (features[2] <= 0.003244615043513477) {
                            if (features[0] <= -0.03575599752366543) {
                                classes[0] = 330;
                                classes[1] = 39;
                            } else {
                                classes[0] = 360;
                                classes[1] = 138;
                            }
                        } else {
                            if (features[2] <= 0.01407763920724392) {
                                classes[0] = 607;
                                classes[1] = 57;
                            } else {
                                classes[0] = 2837;
                                classes[1] = 48;
                            }
                        }
                    } else {
                        if (features[2] <= -0.021189325489103794) {
                            if (features[0] <= 0.008428129367530346) {
                                classes[0] = 113;
                                classes[1] = 342;
                            } else {
                                classes[0] = 82;
                                classes[1] = 90;
                            }
                        } else {
                            if (features[2] <= 0.010304238181561232) {
                                classes[0] = 290;
                                classes[1] = 266;
                            } else {
                                classes[0] = 142;
                                classes[1] = 28;
                            }
                        }
                    }
                } else {
                    if (features[2] <= 0.00489223818294704) {
                        if (features[1] <= 0.4410252124071121) {
                            if (features[0] <= -0.034201690927147865) {
                                classes[0] = 1971;
                                classes[1] = 653;
                            } else {
                                classes[0] = 2273;
                                classes[1] = 234;
                            }
                        } else {
                            if (features[3] <= 1.035286784172058) {
                                classes[0] = 1973;
                                classes[1] = 195;
                            } else {
                                classes[0] = 9665;
                                classes[1] = 144;
                            }
                        }
                    } else {
                        if (features[2] <= 0.016115683130919933) {
                            if (features[1] <= 0.4877214878797531) {
                                classes[0] = 2990;
                                classes[1] = 137;
                            } else {
                                classes[0] = 1448;
                                classes[1] = 18;
                            }
                        } else {
                            if (features[0] <= -0.04386013746261597) {
                                classes[0] = 7991;
                                classes[1] = 130;
                            } else {
                                classes[0] = 7008;
                                classes[1] = 44;
                            }
                        }
                    }
                }
            }
        }

        return findMax(classes);
    };

};

function predict(NBSI, NDVI, NDVI_RE3, CL_RE){
    var clf = new DecisionTreeClassifier();
    return [1/(1+Math.exp(-1*clf.predict([NBSI, NDVI, NDVI_RE3, CL_RE])))];
}
"""

evalscripts_fapar = """
//VERSION=3 (auto-converted from 2)
var degToRad = Math.PI / 180;

function evaluatePixelOrig(samples) {
  var sample = samples[0];
  var b03_norm = normalize(sample.B03, 0, 0.253061520471542);
  var b04_norm = normalize(sample.B04, 0, 0.290393577911328);
  var b05_norm = normalize(sample.B05, 0, 0.305398915248555);
  var b06_norm = normalize(sample.B06, 0.006637972542253, 0.608900395797889);
  var b07_norm = normalize(sample.B07, 0.013972727018939, 0.753827384322927);
  var b8a_norm = normalize(sample.B8A, 0.026690138082061, 0.782011770669178);
  var b11_norm = normalize(sample.B11, 0.016388074192258, 0.493761397883092);
  var b12_norm = normalize(sample.B12, 0, 0.493025984460231);
  var viewZen_norm = normalize(Math.cos(sample.viewZenithMean * degToRad), 0.918595400582046, 1);
  var sunZen_norm  = normalize(Math.cos(sample.sunZenithAngles * degToRad), 0.342022871159208, 0.936206429175402);
  var relAzim_norm = Math.cos((sample.sunAzimuthAngles - sample.viewAzimuthMean) * degToRad)

  var n1 = neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
  var n2 = neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
  var n3 = neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
  var n4 = neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);
  var n5 = neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm);

  var l2 = layer2(n1, n2, n3, n4, n5);

  var fapar = denormalize(l2, 0.000153013463222, 0.977135096979553);
  return {
    default: [fapar]
  }
}

function neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum =
	- 0.887068364040280
	+ 0.268714454733421 * b03_norm
	- 0.205473108029835 * b04_norm
	+ 0.281765694196018 * b05_norm
	+ 1.337443412255980 * b06_norm
	+ 0.390319212938497 * b07_norm
	- 3.612714342203350 * b8a_norm
	+ 0.222530960987244 * b11_norm
	+ 0.821790549667255 * b12_norm
	- 0.093664567310731 * viewZen_norm
	+ 0.019290146147447 * sunZen_norm
	+ 0.037364446377188 * relAzim_norm;

  return tansig(sum);
}

function neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum =
	+ 0.320126471197199
	- 0.248998054599707 * b03_norm
	- 0.571461305473124 * b04_norm
	- 0.369957603466673 * b05_norm
	+ 0.246031694650909 * b06_norm
	+ 0.332536215252841 * b07_norm
	+ 0.438269896208887 * b8a_norm
	+ 0.819000551890450 * b11_norm
	- 0.934931499059310 * b12_norm
	+ 0.082716247651866 * viewZen_norm
	- 0.286978634108328 * sunZen_norm
	- 0.035890968351662 * relAzim_norm;

  return tansig(sum);
}

function neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum =
	+ 0.610523702500117
	- 0.164063575315880 * b03_norm
	- 0.126303285737763 * b04_norm
	- 0.253670784366822 * b05_norm
	- 0.321162835049381 * b06_norm
	+ 0.067082287973580 * b07_norm
	+ 2.029832288655260 * b8a_norm
	- 0.023141228827722 * b11_norm
	- 0.553176625657559 * b12_norm
	+ 0.059285451897783 * viewZen_norm
	- 0.034334454541432 * sunZen_norm
	- 0.031776704097009 * relAzim_norm;

  return tansig(sum);
}

function neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum =
	- 0.379156190833946
	+ 0.130240753003835 * b03_norm
	+ 0.236781035723321 * b04_norm
	+ 0.131811664093253 * b05_norm
	- 0.250181799267664 * b06_norm
	- 0.011364149953286 * b07_norm
	- 1.857573214633520 * b8a_norm
	- 0.146860751013916 * b11_norm
	+ 0.528008831372352 * b12_norm
	- 0.046230769098303 * viewZen_norm
	- 0.034509608392235 * sunZen_norm
	+ 0.031884395036004 * relAzim_norm;

  return tansig(sum);
}

function neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm) {
  var sum =
	+ 1.353023396690570
	- 0.029929946166941 * b03_norm
	+ 0.795804414040809 * b04_norm
	+ 0.348025317624568 * b05_norm
	+ 0.943567007518504 * b06_norm
	- 0.276341670431501 * b07_norm
	- 2.946594180142590 * b8a_norm
	+ 0.289483073507500 * b11_norm
	+ 1.044006950440180 * b12_norm
	- 0.000413031960419 * viewZen_norm
	+ 0.403331114840215 * sunZen_norm
	+ 0.068427130526696 * relAzim_norm;

  return tansig(sum);
}

function layer2(neuron1, neuron2, neuron3, neuron4, neuron5) {
  var sum =
	- 0.336431283973339
	+ 2.126038811064490 * neuron1
	- 0.632044932794919 * neuron2
	+ 5.598995787206250 * neuron3
	+ 1.770444140578970 * neuron4
	- 0.267879583604849 * neuron5;

  return sum;
}

function normalize(unnormalized, min, max) {
  return 2 * (unnormalized - min) / (max - min) - 1;
}
function denormalize(normalized, min, max) {
  return 0.5 * (normalized + 1) * (max - min) + min;
}
function tansig(input) {
  return 2 / (1 + Math.exp(-2 * input)) - 1;
}

function setup() {
  return {
    input: [{
      bands: [
          "B03",
          "B04",
          "B05",
          "B06",
          "B07",
          "B8A",
          "B11",
          "B12",
          "viewZenithMean",
          "viewAzimuthMean",
          "sunZenithAngles",
          "sunAzimuthAngles"
      ]
    }],
    output: [
        {
          id: "default",
          sampleType: "AUTO",
          bands: 1
        }
    ]
  }
}


function evaluatePixel(sample, scene, metadata, customData, outputMetadata) {
  const result = evaluatePixelOrig([sample], [scene], metadata, customData, outputMetadata);
  return result[Object.keys(result)[0]];
}
"""

evalscript_ndvi = """
//VERSION=3
function setup() {
    return {
    input: [
      {
        bands: ["B04", "B08", "dataMask"],
      }
    ],
    output: [
      {
        id: "bands",
        bands: ["NDVI"],
        sampleType: SampleType.FLOAT32
      },
      {
        id: "dataMask",
        bands: 1
      }
    ]    };
}

function evaluatePixel(samples) {
    return {
      bands: [index(samples.B08, samples.B04)],
      dataMask: [samples.dataMask]
    };
}
"""
