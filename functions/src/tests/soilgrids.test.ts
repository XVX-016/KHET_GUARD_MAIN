import { parseSoilGridsResponse } from '../services/soilgrids';

describe('SoilGrids parser', () => {
  it('extracts features from common shapes', () => {
    const data = {
      properties: {
        phh2o: { mean: 6.5 },
        orgc: { mean: 1.2 },
        clay: { mean: 22 },
        sand: { mean: 45 }
      }
    };
    const res = parseSoilGridsResponse(data);
    expect(res.ph).toBe(6.5);
    expect(res.soc).toBe(1.2);
    expect(res.clayPercent).toBe(22);
    expect(res.sandPercent).toBe(45);
  });
});

