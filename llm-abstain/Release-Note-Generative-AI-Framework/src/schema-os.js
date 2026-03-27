/**
 * Data schema for OS components from ReleaseTrain.
 * API: https://releasetrain.io/api/component?q=os
 *
 * In versionSearchTags, the first value of the array contains the "os" keyword.
 * Example: ["os", "android", "production", "2025[29]0213", "16.0.0"]
 */
export const OS_COMPONENT_SCHEMA = {
  _id: String,
  versionId: String,
  versionNumber: String,           // Main display: use as-is or "slice the last value"
  versionProductBrand: String,
  versionProductName: String,       // e.g. "Android"
  versionProductType: String,       // "OS"
  versionProductLicense: String,   // Additional info, e.g. "Apache 2.0"
  versionReleaseChannel: String,
  versionReleaseNotes: String,      // Additional info, URL
  versionReleaseDate: String,
  versionTimestamp: Number,
  versionReleaseComments: String,
  versionVerfied: String,
  versionReleaseTags: Array,
  versionSearchTags: Array,        // First value = "os", e.g. ["os","android","production","2025[29]0213","16.0.0"]
  versionTimestampLastUpdate: String,
  versionPredictedComponentType: String,
  classification: Object,
};

/**
 * Shape response for "What is the version of OS Android on 2-02-2026?"
 * - Main: versionNumber (last value, e.g. last segment or full "16.0.0")
 * - Additional: versionReleaseNotes, versionProductLicense
 */
export function formatOSResponse(doc) {
  if (!doc) return null;
  const versionNumber = doc.versionNumber ?? '';
  // Main display: versionNumber (full). "Slice the last value" = optional last segment, e.g. "16.0.0" -> "0"
  const segments = versionNumber.split('.');
  const lastValue = segments.length > 1 ? segments.pop() : versionNumber;
  return {
    main: versionNumber,
    lastValue,
    versionNumber: versionNumber,
    additional: {
      versionReleaseNotes: doc.versionReleaseNotes ?? '',
      versionProductLicense: doc.versionProductLicense ?? '',
    },
    raw: {
      versionProductName: doc.versionProductName,
      versionReleaseDate: doc.versionReleaseDate,
      versionSearchTags: doc.versionSearchTags,
    },
  };
}
