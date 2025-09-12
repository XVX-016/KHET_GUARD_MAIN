import { recommendCrops } from '../services/recommender';

test('recommender returns top 3', () => {
  const features: any = {
    lat: 0,
    lon: 0,
    soil: { ph: 6.5, clayPercent: 20, sandPercent: 40 },
    weather: { sevenDayRainfallMm: 30, sevenDayAvgTempC: 25 },
    ndvi: { medianMonthlyNdvi: [0.3, 0.4, 0.5] }
  };
  const res = recommendCrops(features);
  expect(res).toHaveLength(3);
  expect(res[0]).toHaveProperty('crop');
  expect(res[0]).toHaveProperty('score');
});

