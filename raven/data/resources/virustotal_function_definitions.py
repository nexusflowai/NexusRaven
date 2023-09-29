def vt_get_dns_resolution_object(id: str, x_apikey: str):
    """
    This endpoint retrieves a Resolution object by its ID. A resolution object ID is made by appending the IP and the domain it resolves to together.

    Domain-IP resolutions. Resolution objects include the following attributes:
    date: <integer> date when the resolution was made (UTC timestamp).
    host_name: <string> domain or subdomain requested to the resolver.
    host_name_last_analysis_stats: <dictionary> last detection stats from the resolution's domain. Similar to the domains's last_analysis_stats attribute.
    ip_address: <string> IP address the domain was resolved to.
    ip_address_last_analysis_stats: <dictionary> last detection stats from the resolution's IP address. Similar to the IP address' last_analysis_stats attribute.
    resolver: <string> source of the resolution.

    Args:
    - id: string, required, Resolution object ID
    - x-apikey: string, required, Your API key
    """


def vt_get_objects_related_to_ip_address(
    ip: str, relationship: str, x_apikey: str, limit: int = None, cursor: str = None
):
    """
    IP addresses have number of relationships to other objects. This returns ALL objects that fit the relationship.

    The relationships are documented here:
    - comments: The comments for the IP address. Returns a list of comments.
    - communicating_files: Files that communicate with the IP address. Returns a list of files.
    - downloaded_files: Files downloaded from the IP address. VT Enterprise users only. Returns a list of files.
    - graphs: Graphs including the IP address. Returns a list of graphs.
    - historical_ssl_certificates: SSL certificates associated with the IP. Returns a list of SSL certificates.
    - historical_whois: WHOIS information for the IP address. Retrurns a list of Whois attributes.
    - related_comments: Community posted comments in the IP's related objects. Returns a list of comments.
    - related_references: Returns the references related to the IP address. Returns a list of References.
    - related_threat_actors: Threat actors related to the IP address. Returns a list of threat actors.
    - referrer_files: Files containing the IP address. Returns a list of Files.
    - resolutions: Resolves the IP addresses. Returns a list of resolutions.
    - urls: Returns a list of URLs related to the IP address. Returns a list of URLs.

    Args:
    - ip, string, required, IP address
    - relationship, string, required, Relationship name (see the list of items from above)
    - x-apikey, string, required, Your API key
    - limit, int32, optional, Maximum number of comments to retrieve
    - cursor, string, optional, Continuation cursor
    """


def vt_get_ip_address_report(ip: str, x_apikey: str):
    """
    Retrieve an IP address report. These reports condense all of the recent activity that VirusTotal has seen for the resource under consideration, as well as contextual information about it.
    This function specifically generates these reports using the IP address parameter.

    Args:
    - ip: string, required, IP address
    - x-apikey: string, required, Your API key
    """


def vt_add_votes_to_ip_address(ip: str, data: dict, x_apikey: str):
    """
    With this function you can post a vote for a given file. The body for the POST request must be the JSON representation of a vote object. Note however that you don't need to provide an ID for the object, as they are automatically generated for new votes. The verdict attribute must have be either harmless or malicious.

    Please ensure that the JSON object you provide conforms accurately to valid JSON standards.

    Args:
    - ip, string, required, IP address
    - data, json, Vote object
    - x-apikey, string, required, Your API key
    """


def vt_get_domain_report(domain: str, x_apikey: str):
    """
    Retrieves a domain report. These reports contain information regarding the domain itself that VirusTotal has collected.

    Args:
    - domain: string, required, Domain name
    - x-apikey: string, required, Your API key
    """


def vt_get_comments_on_ip_address(
    ip: str, x_apikey: str, limit: int = None, cursor: str = None
):
    """
    Retrieves the comments on a provided IP address. Returns a list of Comment objects.

    Args:
    - ip, string, required, IP address
    - x-apikey, string, required, Your API key
    - limit, int32, optional, Maximum number of comments to retrieve
    - cursor, string, optional, Continuation cursor
    """


def vt_add_comment_to_ip_address(ip: str, data: dict, x_apikey: str):
    """
    With this function you can post a comment for a given IP address. The body for the POST request must be the JSON representation of a comment object. Notice however that you don't need to provide an ID for the object, as they are automatically generated for new comments.
    However, please note that you will need to provide a valid data JSON for using this function.

    Any word starting with # in your comment's text will be considered a tag, and added to the comment's tag attribute.

    Returns a Comment object.

    Args:
    - ip: string, required, IP address
    - data: json, required, A comment object
    - x-apikey: string, required, Your API key
    """


def vt_get_object_descriptors_related_to_ip_address(
    ip: str, relationship: str, x_apikey: str, limit: int = None, cursor: str = None
):
    """
    This specifically returns related object's IDs (and context attributes, if any). Please note that this will not return all attributes.

    You are expected to provide the relationship to the object you're interested in. The valid relationships are as follows.

    The relationships are documented here:
    - comments: The comments for the IP address.
    - communicating_files: Files that communicate with the IP address.
    - downloaded_files: Files downloaded from the IP address. VT Enterprise users only.
    - graphs: Graphs including the IP address.
    - historical_ssl_certificates: SSL certificates associated with the IP.
    - historical_whois: WHOIS information for the IP address. Retrurns a list of Whois attributes.
    - related_comments: Community posted comments in the IP's related objects.
    - related_references: Returns the references related to the IP address.
    - related_threat_actors: Threat actors related to the IP address.
    - referrer_files: Files containing the IP address.
    - resolutions: Resolves the IP addresses.
    - urls: Returns a list of URLs related to the IP address.

    Here are some useful descriptions of the arguments in this API, with the format - name of this argument: type of the data, required or optional, description of this argument.
    - ip: string, required, IP address
    - relationship: string, required, Relationship name (see table)
    - x-apikey: string, required, Your API key
    - limit: int32, optional, Maximum number of comments to retrieve
    - cursor: string, optional, Continuation cursor
    """


def vt_get_objects_related_to_domain(
    domain: str, relationship: str, x_apikey: str, limit: int = None, cursor: str = None
):
    """
    Objects are a key concept in the VirusTotal API. Each object has an identifier and a type.
    Each object has an associated URL, and each domain is associated with objects.
    This function returns ALL of the objects related to the domain, based on the specified relationship.

    The following describe the valid relationship:
    - caa_records: Records CAA for the domain.
    - cname_records: Records CNAME for the domain.
    - comments: Community posted comments about the domain.
    - communicating_files: Files that communicate with the domain.
    - downloaded_files: Files downloaded from that domain.
    - graphs: All graphs that include the domain.
    - historical_ssl_certificates: SSL certificates associated with the domain.
    - historical_whois: WHOIS information for the domain.
    - immediate_parent: Domain's immediate parent.
    - mx_records: Records MX for the domain.
    - ns_records: Records NS for the domain.
    - parent: Domain's top parent.
    - referrer_files: Refers to any and all files that contain this domain.
    - related_comments: Community posted comments in the domain's related objects.
    - related_references: Refers to the References related to the domain.
    - related_threat_actors: Refers to the threat actors related to the domain. A list of Threat Actors.
    - resolutions: DNS resolutions for the domain.
    - soa_records: Records SOA for the domain.
    - siblings: Refers to the Domain's sibling domains.
    - subdomains: Refers to the Domain's subdomains.
    - urls: Refers to the URLs that contain this domain.
    - user_votes: Refers to the current user's votes.


    Args:
    - domain: string, required, Domain name
    - relationship, string, required, Relationship name (see table)
    - x-apikey, string, required, Your API key
    - limit, int32, optional, Maximum number of comments to retrieve
    - cursor, string, optional, Continuation cursor
    """


def vt_get_object_descriptors_related_to_domain(
    domain: str, relationship: str, x_apikey: str, limit: int = None, cursor: str = None
):
    """
    This specifically returns related object's IDs (and context attributes, if any). Please note that this will not return all attributes. This will return objects relating to a domain.

    - caa_records: Records CAA for the domain.
    - cname_records: Records CNAME for the domain.
    - comments: Community posted comments about the domain.
    - communicating_files: Files that communicate with the domain.
    - downloaded_files: Files downloaded from that domain.
    - graphs: All graphs that include the domain.
    - historical_ssl_certificates: SSL certificates associated with the domain.
    - historical_whois: WHOIS information for the domain.
    - immediate_parent: Domain's immediate parent.
    - mx_records: Records MX for the domain.
    - ns_records: Records NS for the domain.
    - parent: Domain's top parent.
    - referrer_files: Refers to any and all files that contain this domain.
    - related_comments: Community posted comments in the domain's related objects.
    - related_references: Refers to the References related to the domain.
    - related_threat_actors: Refers to the threat actors related to the domain. A list of Threat Actors.
    - resolutions: DNS resolutions for the domain.
    - soa_records: Records SOA for the domain.
    - siblings: Refers to the Domain's sibling domains.
    - subdomains: Refers to the Domain's subdomains.
    - urls: Refers to the URLs that contain this domain.
    - user_votes: Refers to the current user's votes.

    Args:
    - domain: string, required, Domain name
    - relationship: string, required, Relationship name (see table)
    - x-apikey: string, required, Your API key
    - limit: int32, optional, Maximum number of comments to retrieve
    - cursor: string, optional, Continuation cursor
    """


def vt_get_comments_on_domain(
    domain: str, x_apikey: str, limit: int = None, cursor: str = None
):
    """
    This function will retrieve comments on a specified domain.

    Args:
    - domain, string, required, Domain name
    - x-apikey, string, required, Your API key
    - limit, int32, optional, Maximum number of comments to retrieve
    - cursor, string, optional, Continuation cursor
    """


def vt_get_votes_on_ip_address(ip: str):
    """
    This function will retrieve votes on a provided IP address.

    Args:
    - ip: string, required, ip address
    """
