import requests
import re
import json
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Constants
BASE_URL = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
URL_RT = 'https://releasetrain.io/api/v'

# Calculate dates for the last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=1)

# Format dates to ISO 8601
start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

print(end_date_str)

# List of known brands
KNOWN_BRANDS = [
    "Adobe", "Amazon", "Apple", "Cisco", "Dell", "Facebook", "GitHub", "Google", "HP", "Huawei",
    "IBM", "Intel", "Jenkins", "LinkedIn", "Microsoft", "Mozilla", "NetApp", "Nokia", "Nvidia",
    "Oracle", "Qualcomm", "Red Hat", "Samsung", "Siemens", "Sony", "VMware", "Zoom", 
    "Cisco Systems", "Fortinet", "Palo Alto Networks", "Sophos", "Symantec", "Check Point",
    "Juniper Networks", "McAfee", "Kaspersky", "Trend Micro", "F5 Networks", "Radware",
    "FireEye", "Aruba Networks", "Barracuda Networks", "SonicWall", "CrowdStrike", "Rapid7",
    "Proofpoint", "SentinelOne", "Zscaler", "Darktrace", "Tenable", "Splunk", "Elastic",
    "Bitdefender", "CyberArk", "Okta", "OneLogin", "ForgeRock", "Ping Identity", "Duo Security",
    "Sumo Logic", "Varonis", "Cyberreason", "Mimecast", "Forcepoint", "Sophos", "Citrix",
    "Blue Coat", "RSA", "VMware Carbon Black", "Avast", "ESET", "Commvault", "Veritas", 
    "LogRhythm", "AlienVault", "Trustwave", "Corelight", "Devo", "Infoblox", "Skybox Security",
    "Vectra AI", "ZeroFOX"
]

# List of known products
KNOWN_PRODUCTS = [
    "WordPress", "git.kernel.org", "packetstormsecurity", "metasploit", "nmap", "wireshark",
    "Firefox", "Chrome", "Safari", "Edge", "Opera", "Internet Explorer", "Photoshop", "Illustrator",
    "Premiere Pro", "After Effects", "InDesign", "Lightroom", "Acrobat", "Flash Player",
    "Windows", "Windows Server", "Linux", "macOS", "iOS", "Android", "Ubuntu", "Fedora", 
    "Debian", "CentOS", "RHEL", "Arch Linux", "Kali Linux", "Mint", "Elementary OS", "Zorin OS", 
    "Alpine Linux", "Gentoo", "Slackware", "OpenSUSE", "FreeBSD", "NetBSD", "OpenBSD", 
    "Solaris", "AIX", "HP-UX", "Tru64", "IRIX", "SCO Unix", "UNIX System V", "Cygwin", 
    "WSL", "Docker", "Kubernetes", "OpenStack", "CloudStack", "VMware vSphere", "Proxmox",
    "Hyper-V", "Xen", "VirtualBox", "QEMU", "Libvirt", "KVM", "AWS", "Azure", "Google Cloud",
    "Alibaba Cloud", "IBM Cloud", "Oracle Cloud", "Salesforce", "SAP", "ServiceNow", "Workday",
    "Slack", "Zoom", "Teams", "Skype", "Webex", "GoToMeeting", "BlueJeans", "Chime", 
    "Adobe Connect", "Citrix", "HCL Notes", "Lotus Domino", "SharePoint", "OneDrive", "Google Drive",
    "Dropbox", "Box", "Nextcloud", "ownCloud", "Seafile", "Syncthing", "Resilio Sync", "Mega",
    "Tresorit", "pCloud", "Sync.com", "Egnyte", "Amazon S3", "Google Cloud Storage", "Azure Blob Storage",
    "Backblaze B2", "Wasabi", "DigitalOcean Spaces", "Linode Object Storage", "Scality", 
    "Caringo", "Minio", "Ceph", "GlusterFS", "Hadoop HDFS", "Amazon EFS", "Google Filestore",
    "Azure Files", "NetApp ONTAP", "Dell EMC Isilon", "IBM Spectrum Scale", "Qumulo", 
    "WekaIO", "Panzura", "Nasuni", "LucidLink", "Ctera", "Komprise", "PeerGFS", "Lustre", 
    "BeeGFS", "OrangeFS", "Panasas", "StorNext", "ZFS", "Btrfs", "XFS", "JFS", "Ext4", 
    "ReiserFS", "F2FS", "NTFS", "FAT32", "exFAT", "HFS+", "APFS", "NFS", "SMB", "CIFS", 
    "AFP", "iSCSI", "Fibre Channel", "Infiniband", "RoCE", "NVMe", "SATA", "SAS", "PCIe", 
    "USB", "Thunderbolt", "FireWire", "IEEE 1394", "Bluetooth", "Wi-Fi", "Ethernet", 
    "Token Ring", "FDDI", "ARCnet", "ATM", "Frame Relay", "MPLS", "SONET", "SDH", 
    "ISDN", "POTS", "xDSL", "DOCSIS", "Satellite", "Cellular", "5G", "4G", "3G", "2G", 
    "LoRa", "Sigfox", "NB-IoT", "Zigbee", "Z-Wave", "EnOcean", "KNX", "BACnet", "Modbus", 
    "PROFINET", "EtherCAT", "Powerline", "IEEE 1901", "HomePlug", "G.hn", "MoCA", "PoE",
    "PON", "FTTH", "FTTP", "FTTB", "FTTC", "FTTN", "DSLAM", "OLT", "ONT", "ONU", 
    "MW", "VSAT", "NGSO", "LEO", "MEO", "GEO", "Pico", "Femto", "Micro", "Macro", 
    "Metro", "Core", "Access", "Edge", "Fog", "Cloud", "Enterprise", "Data Center", 
    "Campus", "Branch", "WAN", "MAN", "LAN", "PAN", "NAN", "HAN", "CAN", "BAN", "RAN", 
    "SAN", "VLAN", "VPN", "MPLS VPN", "IPsec", "SSL VPN", "TLS VPN", "DTLS VPN", "GRE", 
    "L2TP", "PPTP", "IKE", "IKEv2", "DVPN", "VXLAN", "NVGRE", "STT", "GTP", "SR-IOV", 
    "OVS", "DPDK", "SPDK", "VPP", "XDP", "AF_XDP", "eBPF", "tc", "iproute2", "netfilter", 
    "iptables", "nftables", "firewalld", "ufw", "shorewall", "pfSense", "OPNsense", "Endian",
    "VyOS", "MikroTik", "Ubiquiti", "TP-Link", "D-Link", "Netgear", "Belkin", "Linksys", 
    "Cisco Meraki", "Arista", "Juniper", "Palo Alto Networks", "Fortinet", "Check Point", 
    "Sophos", "F5", "A10", "Radware", "Citrix ADC", "Barracuda", "SonicWall", "WatchGuard", 
    "FireEye", "CrowdStrike", "Proofpoint", "Mimecast", "Cylance", "SentinelOne", "Carbon Black", 
    "McAfee", "Symantec", "Norton", "Kaspersky", "Trend Micro", "Bitdefender", "ESET", 
    "Avast", "AVG", "Sophos", "Panda", "Webroot", "Malwarebytes", "F-Secure", "G Data", 
    "Dr.Web", "Comodo", "PC-cillin", "ZoneAlarm", "Ad-Aware", "Spybot", "HijackThis", 
    "Stinger", "RogueKiller", "HitmanPro", "Zemana", "Emsisoft", "IObit", "Glarysoft", 
    "CCleaner", "BleachBit", "Wise", "Advanced SystemCare", "Auslogics", "Revo", "Geek", 
    "IOBit", "AVG", "McAfee", "Norton", "Panda", "PC-cillin", "Sophos", "Trend Micro", 
    "Webroot", "Ad-Aware", "Spybot", "Malwarebytes", "Microsoft Defender", "Microsoft Security Essentials"
]

def fetch_cve_data(start_date, end_date):
    url = f"{BASE_URL}?pubStartDate={start_date}&pubEndDate={end_date}"
    print(url)
    headers = {
        'Content-Type': 'application/json'
    }
    # print(f"Fetching CVE data from {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # print("Data fetched successfully.")
        return response.json()
    else:
        # print(f"Failed to fetch data. Status code: {response.status_code}")
        # print(f"Response: {response.text}")
        return None

# Fix the process_cve_entries function to extract the correct product name from references

def process_cve_entries(cve_data):
    if not cve_data or 'vulnerabilities' not in cve_data:
        # print("No CVE data to process.")
        return

    for item in cve_data['vulnerabilities']:
        cve = item['cve']
        references = cve.get('references', [])
        if references:
            url = references[0].get('url', '')
        else:
            url = ''
        product_name = extract_product_name_from_notes(cve.get('descriptions', [{}])[0].get('value', ''), url)
        product_brand = extract_product_name_from_notes(cve.get('descriptions', [{}])[0].get('value', ''), url)

        v = {
            'versionId': '',
            'versionNumber': '',
            'versionProductBrand': product_brand,
            'versionProductName': product_name,
            'versionProductType': '',
            'versionProductLicense': '',
            'versionReleaseChannel': 'cve',
            'versionReleaseComments': cve.get('sourceIdentifier', ''),
            'versionReleaseDate': datetime.strptime(cve.get('published', ''), '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d'),
            'versionTimestamp': '',
            'versionReleaseNotes': cve.get('descriptions', [{}])[0].get('value', ''),
            'versionVerfied': '',
            'versionReleaseTags': [],
            'versionSearchTags': [],
            'versionStatus': cve.get('vulnStatus', ''),
            'versionUrl': f"https://nvd.nist.gov/vuln/detail/{cve.get('id', '')}"
        }

        if not v['versionProductName']:
            # print("Skipping CVE entry due to missing product name.")
            continue

        v['versionNumber'] = extract_version_number(v['versionReleaseNotes'])

        if not v['versionNumber']:
            # print("Skipping CVE entry due to missing version number.")
            continue

        v['versionId'] = f"{v['versionReleaseDate']}{v['versionProductName']}{v['versionNumber']}"

        # print(json.dumps(v, indent=4))

        #print(f"Posting version info for {v['versionProductName']} version {v['versionNumber']} to Release Train API.")
        response = requests.post(URL_RT, data=json.dumps(v), headers={'Content-type': 'application/json'})
        print(f"Response from Release Train API: {response.status_code} {v['versionProductName']}")

def extract_product_name_from_notes(notes, url):
    #print(f"Extracting product name from notes: {notes}")

    # Check for pattern " in " followed by words starting with uppercase letters
    match = re.search(r' in ((?:[A-Z][a-zA-Z]*\s*)+)', notes)
    if match:
        return match.group(1).strip()

    # Check for known brands and products
    for brand in KNOWN_BRANDS:
        if re.search(r'\b' + re.escape(brand.lower()) + r'\b', notes.lower()):
            return brand
    for product in KNOWN_PRODUCTS:
        if re.search(r'\b' + re.escape(product.lower()) + r'\b', notes.lower()):
            return product

    # If no known brand or product is found, use the URL subdomain/domain
    #print(f"Unknown product in notes. Extracting from URL: {url}")
    return extract_product_name(url)

def extract_product_name(url):
    #print(f"Extracting product name from URL: {url}")
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    subdomain = domain.split('.')
    product_name = domain
    if len(subdomain) > 2:
        product_name = '.'.join(subdomain[-3:])
    elif len(subdomain) > 1:
        product_name = '.'.join(subdomain[-2:])
    
    for brand in KNOWN_BRANDS:
        if brand.lower() in product_name.lower():
            return brand
    for product in KNOWN_PRODUCTS:
        if product.lower() in product_name.lower():
            return product
    return product_name if re.match('^[\w.-]+$', product_name) else "Mitre CVE"


def extract_version_number(notes):
    # print(f"Extracting version number from notes: {notes}")
    pattern = r'[0-9]+\.[0-9]+(\.[0-9]+)?'
    result = re.search(pattern, notes)
    return result.group(0) if result else None

def main():
    cve_data = fetch_cve_data(start_date_str, end_date_str)
    
    if cve_data:
        # print(f"Retrieved {len(cve_data['vulnerabilities'])} CVE records for the last 7 days.")
        process_cve_entries(cve_data)

if __name__ == '__main__':
    main()
