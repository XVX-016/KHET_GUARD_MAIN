import axios from 'axios';

export type WeatherFeatures = {
  sevenDayRainfallMm: number;
  sevenDayAvgTempC: number;
};

export async function getWeatherFeatures(lat: number, lon: number): Promise<WeatherFeatures> {
  const apiKey = process.env.OPENWEATHER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENWEATHER_API_KEY not set');
  }
  const url = `https://api.openweathermap.org/data/2.5/onecall?lat=${lat}&lon=${lon}&exclude=minutely,hourly,current,alerts&units=metric&appid=${apiKey}`;
  const { data } = await axios.get(url, { timeout: 15000 });
  const daily = data?.daily || [];
  const days = daily.slice(0, 7);
  const sevenDayRainfallMm = days.reduce((sum: number, d: any) => sum + (Number(d.rain) || 0), 0);
  const sevenDayAvgTempC = days.length
    ? days.reduce((sum: number, d: any) => sum + Number(d.temp?.day ?? 0), 0) / days.length
    : 0;
  return { sevenDayRainfallMm, sevenDayAvgTempC };
}

