# Data schema for Linux vendor patch (ReleaseTrain)

**Source for all queries that mention Linux:** the search API below (Android questions are excluded and use the OS component API).  
Example prompts: *What is the patch for Linux on 14-02-2024?* • *What is the patch for Linux on 14-02-2026?* • *Linux on 03-04-2026?*

**API:** [https://releasetrain.io/api/v/search?q=linux&channel=patch&limit=25&page=1](https://releasetrain.io/api/v/search?q=linux&channel=patch&limit=25&page=1)

- **Unique key:** `versionSearchTags` — e.g. `["linux", "patch", "20260214", "1.1.1"]`
- **Main data:** last value in `versionSearchTags` (e.g. `"1.1.1"` = patch version)
- **Additional information:** `versionUrl`, `versionReleaseNotes`

## Example document

```json
{
  "_id": "6990a6a46f1f3eed0f426c77",
  "versionId": "20260214Linux1.1.1",
  "versionNumber": "1.1.1",
  "versionProductBrand": "Linux",
  "versionProductName": "Linux",
  "versionProductType": "",
  "versionProductLicense": "",
  "versionReleaseChannel": "patch",
  "versionReleaseComments": "416baaa9-dc9f-4396-8d5f-8c081fb06d67",
  "versionReleaseDate": "20260214",
  "versionTimestamp": 1771087524592,
  "versionReleaseNotes": "In the Linux kernel, the following vulnerability has been resolved:\n\nbonding: fix use-after-free...",
  "versionVerfied": "",
  "versionReleaseTags": [],
  "versionSearchTags": ["linux", "patch", "20260214", "1.1.1"],
  "versionStatus": "Received",
  "versionUrl": "https://nvd.nist.gov/vuln/detail/CVE-2026-23171",
  "isCve": true,
  "versionTimestampLastUpdate": "2026-02-14T16:45:24.592Z",
  "classification": {
    "securityType": ["SECURITY"],
    "breakingType": ["Breaking Update", "Critical Failure"],
    "componentType": ["OS"]
  }
}
```

## Prompt → Response

**Prompt:** *What is the patch for Linux on 14-02-2026?*

- **versionSearchTags** (logical key): `["linux", "patch", "20260214", "1.1.1"]`  
  Date in query (14-02-2026) is normalized to `versionReleaseDate` form `20260214` (YYYYMMDD).

**Main data:** last value of `versionSearchTags` → e.g. `"1.1.1"` (patch version).

**Additional information:**

1. `versionUrl`
2. `versionReleaseNotes`

## API response shape (this app)

- **main / version:** last value of `versionSearchTags` (patch version string)
- **additional.versionUrl**
- **additional.versionReleaseNotes**
- **vendor:** e.g. `versionProductName` ("Linux")

The backend detects Linux patch prompts (e.g. “patch for Linux on &lt;date&gt;”), calls the ReleaseTrain search API above, finds the item matching the date (or closest), and returns the above structure on `POST /answer`. The response includes `sourceUrl` with the API URL so clients know the data source for Linux queries.
