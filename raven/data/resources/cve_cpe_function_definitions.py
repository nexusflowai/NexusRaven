def searchCVE(
    cpeName=None,
    cveId=None,
    cvssV2Metrics=None,
    cvssV2Severity=None,
    cvssV3Metrics=None,
    cvssV3Severity=None,
    cweId=None,
    hasCertAlerts=None,
    hasCertNotes=None,
    hasKev=None,
    hasOval=None,
    isVulnerable=None,
    keywordExactMatch=None,
    keywordSearch=None,
    lastModStartDate=None,
    lastModEndDate=None,
    noRejected=None,
    pubStartDate=None,
    pubEndDate=None,
    sourceIdentifier=None,
    versionEnd=None,
    versionEndType=None,
    versionStart=None,
    versionStartType=None,
    virtualMatchString=None,
    limit=None,
    delay=None,
    key=None,
    verbose=None,
):
    """
    Build and send GET request then return list of objects containing a collection of CVEs. For more information on the parameters available, please visit https://nvd.nist.gov/developers/vulnerabilities

    Args:
    cpeName (str): Please do not confuse this with keywordSearch; this requires the argument to start with "cpe", whereas the keywordSearch argument allows for arbitrary keywords. This value will be compared agains the CPE Match Criteria within a CVE applicability statement. (i.e. find the vulnerabilities attached to that CPE). Partial match strings are allowed.

    cveId (str): Please pass in a string integer, like "1" or "30". Returns a single CVE that already exists in the NVD.

    cvssV2Metrics (str): This parameter returns only the CVEs that match the provided CVSSv2 vector string. Either full or partial vector strings may be used. This parameter cannot be used in requests that include cvssV3Metrics.

    cvssV2Severity (str): Find vulnerabilities having a LOW, MEDIUM, or HIGH version 2 severity.

    cvssV3Metrics (str): This parameter returns only the CVEs that match the provided CVSSv3 vector string. Either full or partial vector strings may be used. This parameter cannot be used in requests that include cvssV2Metrics.

    cvssV3Severity (str): Find vulnerabilities having a LOW, MEDIUM, HIGH, or CRITICAL version 3 severity.

    cweId (str): Please pass in a string integer, like "1" or "30". Filter collection by CWE (Common Weakness Enumeration) ID. You can find a list at https://cwe.mitre.org/. A CVE can have multiple CWE IDs assigned to it.

    hasCertAlerts (bool): Returns CVE that contain a Technical Alert from US-CERT.

    hasCertNotes (bool): Returns CVE that contain a Vulnerability Note from CERT/CC.

    hasOval (bool): Returns CVE that contain information from MITRE's Open Vulnerability and Assessment Language (OVAL) before this transitioned to the Center for Internet Security (CIS).

    isVulnerable (bool): Returns CVE associated with a specific CPE, where the CPE is also considered vulnerable. REQUIRES cpeName parameter. isVulnerable is not compatible with virtualMatchString parameter.

    keywordExactMatch (bool): When keywordSearch is used along with keywordExactmatch, it will search the NVD for CVEs containing exactly what was passed to keywordSearch. REQUIRES keywordSearch.

    keywordSearch (str): Searches CVEs where a word or phrase is found in the current description. If passing multiple keywords with a space character in between then each word must exist somewhere in the description, not necessarily together unless keywordExactMatch=True is passed to searchCVE.

    lastModStartDate (str, datetime obj): These parameters return only the CVEs that were last modified during the specified period. If a CVE has been modified more recently than the specified period, it will not be included in the response. If filtering by the last modified date, both lastModStartDate and lastModEndDate are REQUIRED. The maximum allowable range when using any date range parameters is 120 consecutive days.

    lastModEndDate (str, datetime obj): Required if using lastModStartDate.

    noRejected (bool): Filters out all CVEs that are in a reject or rejected status. Searches without this parameter include rejected CVEs.

    pubStartDate (str, datetime obj): These parameters return only the CVEs that were added to the NVD (i.e., published) during the specified period. If filtering by the published date, both pubStartDate and pubEndDate are REQUIRED. The maximum allowable range when using any date range parameters is 120 consecutive days.

    pubEndDate (str, datetime obj): Required if using pubStartDate.

    sourceIdentifier (str): Returns CVE where the data source of the CVE is the value that is passed to sourceIdentifier.

    versionEnd (str): Must be combined with versionEndType and virtualMatchString. Returns only the CVEs associated with CPEs in specific version ranges.

    versionEndType (str): Must be combined with versionEnd and virtualMatchString. Valid values are including or excluding. Denotes to include the specified version in versionEnd, or exclude it.

    versionStart (str): Must be combined with versionStartType and virtualMatchString. Returns only CVEs with specific versions. Requests that include versionStart cannot include a version component in the virtualMatchString.

    versionStartType (str): Must be combined with versionStart and virtualMatchString. Valid values are including or excluding. Denotes to include the specified version in versionStart, or exclude it.

    virtualMatchString (str): A more broad filter compared to cpeName. The cpe match string that is passed to virtualMatchString is compared against the CPE Match Criteria present on CVE applicability statements.

    limit (int): Custom argument to limit the number of results of the search. Allowed any number between 1 and 2000.

    delay (int): Can only be used if an API key is provided. This allows the user to define a delay. The delay must be greater than 0.6 seconds. The NVD API recommends scripts sleep for atleast 6 seconds in between requests.

    key (str): NVD API Key. Allows for the user to define a delay. NVD recommends scripts sleep 6 seconds in between requests. If no valid API key is provided, requests are sent with a 6 second delay.

    verbose (bool): Prints the URL request for debugging purposes.
    """


def searchCPE(
    cpeNameId=None,
    cpeMatchString=None,
    keywordExactMatch=None,
    keywordSearch=None,
    lastModStartDate=None,
    lastModEndDate=None,
    matchCriteriaId=None,
    limit=None,
    key=None,
    delay=None,
    verbose=None,
):
    """
    Build and send GET request then return list of objects containing a collection of CPEs.

    Args:
    cpeNameId (str): Returns a specific CPE record using its UUID. If a correctly formatted UUID is passed but it does not exist, it will return empty results. The UUID is the cpeNameId value when searching CPE.

    cpeMatchString (str): Use a partial CPE name to search for other CPE names.

    keywordExactMatch (bool): Searches metadata within CPE title and reference links for an exact match of the phrase or word passed to it. Must be included with keywordSearch.

    keywordSearch (str): Returns CPE records where a word or phrase is found in the metadata title or reference links. Space characters act as an AND statement.

    lastModStartDate (str/datetime obj): CPE last modification start date. Maximum 120 day range. A start and end date is required. All times are in UTC 00:00. A datetime object or string can be passed as a date. NVDLib will automatically parse the datetime object into the correct format. String Example: ‘2020-06-28 00:00’

    lastModEndDate (str/datetime obj): CPE last modification end date. Maximum 120 day range. Must be included with lastModStartDate. Example: ‘2020-06-28 00:00’

    limit (int): Limits the number of results of the search.

    key (str): NVD API Key. Allows for a request every 0.6 seconds instead of 6 seconds.

    delay (int): Can only be used if an API key is provided. The amount of time to sleep in between requests. Must be a value above 0.6 seconds if an API key is present. delay is set to 6 seconds if no API key is passed.

    verbose (bool): Prints the URL request for debugging purposes.
    """
