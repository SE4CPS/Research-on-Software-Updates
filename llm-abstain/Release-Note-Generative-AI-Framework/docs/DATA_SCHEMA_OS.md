# Data schema for OS (ReleaseTrain)

Data is loaded from:

**API:** [https://releasetrain.io/api/component?q=os](https://releasetrain.io/api/component?q=os)

In `versionSearchTags`, the first value of the dictionary contains the OS keyword.

Example: `["os", "android", "production", "2025[29]0213", "16.0.0"]`

## Example document

```json
{
  "_id": "67bbd8dc537815e5ae106f40",
  "versionId": "2025[29]0213Android16.0.0",
  "versionNumber": "16.0.0",
  "versionProductBrand": "Alphabet",
  "versionProductName": "Android",
  "versionProductType": "OS",
  "versionProductLicense": "Apache 2.0",
  "versionReleaseChannel": "production",
  "versionReleaseNotes": "https://source.android.com/setup/start/build-numbers",
  "versionReleaseDate": "2025[29]0213",
  "versionTimestamp": 1740363996589,
  "versionReleaseComments": "",
  "versionVerfied": "",
  "versionReleaseTags": [],
  "versionSearchTags": ["os", "android", "production", "2025[29]0213", "16.0.0"],
  "versionTimestampLastUpdate": "2025-03-10T22:09:07.851Z",
  "versionPredictedComponentType": "OS",
  "classification": {
    "securityType": ["SECURITY"],
    "breakingType": [],
    "componentType": ["OS", "MOBILE"]
  }
}
```

## Prompt → Response

**Prompt:** *What is the version of OS Android on 2-02-2026?*

**Main data to display:** `versionNumber` — use the full value or slice the last segment (e.g. `16.0.0` → last value `0` if you need only the patch segment).

**Additional information:**

1. `versionReleaseNotes`
2. `versionProductLicense`

## API response shape (this app)

- **main:** `versionNumber` (or last segment)
- **additional.versionReleaseNotes**
- **additional.versionProductLicense**

The backend in `src/server.js` fetches from ReleaseTrain, selects the matching OS document (e.g. Android, optional date), and returns this structure on `GET /api/component?q=os` and `POST /answer`.
