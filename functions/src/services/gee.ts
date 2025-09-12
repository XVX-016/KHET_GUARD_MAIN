import axios from 'axios';

export type NdviSummary = {
  medianMonthlyNdvi: number[]; // last 12 months
  medianMonthlyNdwi?: number[];
};

export async function getNdviSummary(lat: number, lon: number): Promise<NdviSummary> {
  const baseUrl = process.env.GEE_SERVICE_BASE_URL || 'http://localhost:8000';
  const url = `${baseUrl}/ndvi`; // see python service
  try {
    const { data } = await axios.post(url, { lat, lon, buffer_m: 500 }, { timeout: 20000 });
    return {
      medianMonthlyNdvi: data?.medianMonthlyNdvi ?? [],
      medianMonthlyNdwi: data?.medianMonthlyNdwi ?? []
    };
  } catch (_e) {
    return { medianMonthlyNdvi: [], medianMonthlyNdwi: [] };
  }
}

