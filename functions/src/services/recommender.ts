import { NdviSummary } from './gee';
import { SoilFeatures } from './soilgrids';
import { WeatherFeatures } from './openweather';

type Features = {
  lat: number;
  lon: number;
  soil: SoilFeatures;
  weather: WeatherFeatures;
  ndvi: NdviSummary;
};

type Recommendation = { crop: string; score: number; reasons: string[] };

const CANDIDATE_CROPS = [
  'Wheat',
  'Rice',
  'Maize',
  'Sorghum',
  'Millet',
  'Pulses',
  'Cotton',
  'Sugarcane'
];

export function recommendCrops(features: Features): Recommendation[] {
  const { soil, weather, ndvi } = features;
  const results: Recommendation[] = [];
  for (const crop of CANDIDATE_CROPS) {
    const reasons: string[] = [];
    let score = 0;
    // pH scoring
    if (soil.ph !== undefined) {
      const idealPh: Record<string, [number, number]> = {
        Wheat: [6.0, 7.5],
        Rice: [5.5, 7.0],
        Maize: [5.5, 7.5],
        Sorghum: [5.0, 7.5],
        Millet: [5.0, 7.5],
        Pulses: [6.0, 7.5],
        Cotton: [5.8, 8.0],
        Sugarcane: [6.0, 8.0]
      } as const;
      const [lo, hi] = idealPh[crop] || [5.5, 7.5];
      const within = soil.ph >= lo && soil.ph <= hi;
      score += within ? 0.25 : 0.1;
      reasons.push(`pH ${within ? 'ok' : soil.ph < lo ? 'low' : 'high'} (${soil.ph?.toFixed(1)})`);
    }
    // rainfall
    if (weather.sevenDayRainfallMm !== undefined) {
      const idealRain: Record<string, [number, number]> = {
        Wheat: [5, 40],
        Rice: [30, 150],
        Maize: [10, 80],
        Sorghum: [5, 50],
        Millet: [5, 40],
        Pulses: [5, 40],
        Cotton: [10, 70],
        Sugarcane: [40, 200]
      } as const;
      const [rl, rh] = idealRain[crop] || [10, 80];
      const within = weather.sevenDayRainfallMm >= rl && weather.sevenDayRainfallMm <= rh;
      score += within ? 0.25 : 0.1;
      reasons.push(`rainfall ${within ? 'ok' : weather.sevenDayRainfallMm < rl ? 'low' : 'high'} (${weather.sevenDayRainfallMm.toFixed(1)}mm/7d)`);
    }
    // texture via clay/sand
    if (typeof soil.clayPercent === 'number' || typeof soil.sandPercent === 'number') {
      const clay = soil.clayPercent ?? 20;
      const sand = soil.sandPercent ?? 40;
      const loamy = clay >= 10 && clay <= 35 && sand >= 30 && sand <= 60;
      score += loamy ? 0.25 : 0.1;
      reasons.push(`texture ${loamy ? 'loamy-like' : 'extreme'} (clay ${clay}%, sand ${sand}%)`);
    }
    // NDVI trend
    if (ndvi.medianMonthlyNdvi?.length) {
      const len = ndvi.medianMonthlyNdvi.length;
      const first = ndvi.medianMonthlyNdvi[0];
      const last = ndvi.medianMonthlyNdvi[len - 1];
      const trendUp = last >= first;
      score += trendUp ? 0.25 : 0.1;
      reasons.push(`ndvi trend ${trendUp ? 'up' : 'down'}`);
    }
    results.push({ crop, score: Math.min(1, Number(score.toFixed(2))), reasons });
  }
  return results.sort((a, b) => b.score - a.score).slice(0, 3);
}

