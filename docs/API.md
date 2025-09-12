## HTTP API

Base: `https://<region>-<project>.cloudfunctions.net/api`

- POST `/recommendCrop`
  - body: `{ lat: number, lon: number } | { locationId: string }`
  - response: `{ features, recommendations: [{ crop, score, reasons[] }] }`

- POST `/detectPlantDisease`
  - body: `{ imageBase64?: string, storagePath?: string }`
  - response: `{ diagnosis, confidence, suggested[], source }`

- POST `/detectCattle`
  - body: `{ imageBase64?: string, storagePath?: string }`
  - response: `{ breed, confidence, tips[], source }`

- GET `/location/{id}/data`
  - returns cached combined data for a stored location

### Examples

```
http POST $FUNCTIONS_URL/recommendCrop lat:=28.6 lon:=77.2
http POST $FUNCTIONS_URL/detectPlantDisease imageBase64:=@leaf_base64.txt
```
