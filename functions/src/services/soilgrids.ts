import axios from 'axios';

export type SoilFeatures = {
  ph?: number;
  soc?: number;
  clayPercent?: number;
  sandPercent?: number;
};

export async function getSoilFeatures(lat: number, lon: number): Promise<SoilFeatures> {
  const url = `https://rest.soilgrids.org/query?lon=${lon}&lat=${lat}`;
  const { data } = await axios.get(url, { timeout: 15000 });
  return parseSoilGridsResponse(data);
}

export function parseSoilGridsResponse(data: any): SoilFeatures {
  const profile = data?.properties || data?.data || data?.result || {};
  const ph = Number(profile.ph?.mean ?? profile.phh2o?.mean ?? profile.phh2o?.M ?? profile.ph) || undefined;
  const soc = Number(profile.soc?.mean ?? profile.orgc?.mean ?? profile.soc) || undefined;
  const clayPercent = Number(profile.clay?.mean ?? profile.clay) || undefined;
  const sandPercent = Number(profile.sand?.mean ?? profile.sand) || undefined;
  return { ph, soc, clayPercent, sandPercent };
}

