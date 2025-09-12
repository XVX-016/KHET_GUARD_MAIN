import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { getWeatherFeatures } from '../services/openweather';

describe('OpenWeather wrapper', () => {
  const mock = new MockAdapter(axios);
  const old = process.env.OPENWEATHER_API_KEY;
  beforeAll(() => {
    process.env.OPENWEATHER_API_KEY = 'testkey';
  });
  afterAll(() => {
    process.env.OPENWEATHER_API_KEY = old;
    mock.reset();
  });

  it('computes rainfall and avg temp', async () => {
    mock.onGet(/onecall/).reply(200, {
      daily: Array.from({ length: 7 }, (_, i) => ({ rain: i, temp: { day: 20 + i } }))
    });
    const res = await getWeatherFeatures(10, 20);
    expect(res.sevenDayRainfallMm).toBe(21);
    expect(res.sevenDayAvgTempC).toBeCloseTo(23);
  });
});

