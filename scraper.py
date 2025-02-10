import re
from collections import Counter
import tokenizer as t
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
import hashlib

VALID_DOMAINS = ("ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu")

# set of all unqiue URLs identified, might need to change to a dictionary to get urls from each page
TOTAL_URLS = set()

AUTHORITY_COUNT = dict()

PAGE_COUNT = dict()

previousListOfStrings = []


COUNT = 0

def scraper(url, resp):
    links = extract_next_links(url, resp)
    return [link for link in links if is_valid(link)]


def extract_next_links(url, resp):
    # Implementation required. url: the URL that was used to get the page resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there
    # was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    # resp.raw_response.url: the url, again resp.raw_response.content: the content of the page! Return a list with
    # the hyperlinks (as strings) scrapped from resp.raw_response.content
    global COUNT
    print(COUNT)
    COUNT += 1


    if not should_extract(url, resp):  # make sure the URL is valid and is not responding with error
        return list()


    soup = BeautifulSoup(resp.raw_response.content, 'html.parser')
    text = soup.get_text()

    if is_same_content(text):
        print(f'SKIPPED {url}')
        return list()



    urls = parse_urls(resp.raw_response.url, soup)



    valid_urls = []
    for url in urls:
        valid_urls.append(url)
    return valid_urls


def should_extract(url, resp):

    if not is_valid(url):
        return False
    if resp.error:
        return False
    if not is_crawling_allowed(url):
        return False
    if not check_file_size(url, resp):
        return False

    return True

def check_file_size(url, resp):

    minimum_size = 500 #500 bytes
    maximum_size = 35 * 1024 * 1024 #35MB


    if not (minimum_size < len(resp.raw_response.content) < maximum_size):
        print(f'{url}: SIZE INVALID: {len(resp.raw_response.content)} bytes')
        return False
    else:
        #print(f'{url}: SIZE VALID: {len(resp.raw_response.content)} bytes')
        return True


def is_crawling_allowed(url, user_agent="*"):
    # * for all agents
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt" # get url of robots.txt

    rp = RobotFileParser()

    try:
        rp.set_url(robots_url)
        rp.read()  # Read and parse robots.txt
        return rp.can_fetch(user_agent, url)  # Check if crawling is allowed
    except:
        return True  # Assume allowed if there is no robots.txt



def is_duplicate_url(url):
    # checks if urls are valid
    if url not in TOTAL_URLS:
        TOTAL_URLS.add(url)
        return False
    else:
        return True


def is_recursive_url(url, threshold=3):
    # Detects recursive traps where path segments are repeated too many times.
    parsed = urlparse(url)
    path_segments = parsed.path.strip("/").split("/")  # Extract path segments

    # Count occurrences of each segment
    segment_counts = Counter(path_segments)

    # If any segment appears more than the threshold, it's likely a trap
    for segment, count in segment_counts.items():
        if count >= threshold:
            return True  # Recursive pattern detected

    return False  # No recursive pattern


# def is_authority_pass_threshold(authority, threshold=500):
#     if authority in AUTHORITY_COUNT:
#         if AUTHORITY_COUNT[authority] >= threshold: # over the threshold
#             print('count = ', AUTHORITY_COUNT[authority])
#             return True
#         else:
#             AUTHORITY_COUNT[authority] += 1 # add one to occurrences
#             return False
#     else:
#         AUTHORITY_COUNT[authority] = 1 # first appearence
#         return False


def parse_urls(base_url, soup):
    urls = []
    for link in soup.find_all("a"):
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)  # Convert to absolute URL
            defragmented_url = full_url.split('#')[0]
            urls.append(defragmented_url)
    return urls


def _is_valid_authority(url):
    # checks if url authority matches one of the allowed authorities
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()  # Normalize to lowercase for consistency
    # Construct regex dynamically for allowed netlocs
    pattern = r"(" + r"|".join(re.escape(n) for n in VALID_DOMAINS) + r")$"
    return re.search(pattern, netloc) is not None




def is_valid(url):
    # Decide whether to crawl this url or not.
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.

    VALID_DOMAINS = {".ics.uci.edu", ".cs.uci.edu", ".informatics.uci.edu", ".stat.uci.edu"}

    try:
        parsed = urlparse(url)
        # Check if scheme is https or http
        if parsed.scheme not in set(["http", "https"]):
            #print('not in scheme', url)
            return False
        # Check if URL is one of VALID_DOMAINS
        if not _is_valid_authority(url):
            #print('not in domain', url)
            return False
        # Check if URL has already been crawled
        if is_duplicate_url(url):
            #print('duplicate', url)
            return False
        # Check if the path of the URL is recursive
        if is_recursive_url(url):
            #print('recursive', url)
            return False
        #
        # if is_authority_pass_threshold(parsed.netloc):
        #     print('too much authority', url)
        #     return False

        # Get rid of unwanted file types
        return not re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower())

    except TypeError:
        print("TypeError for ", parsed)
        raise


###########################################################
#                 Similarity and Trap Detection
###########################################################

def extract_features(text):
    # Tokenize the text and get word frequencies
    words = tokenize(text)
    frequencies = compute_word_frequencies(words)

    # Create a feature for each word based on its frequency
    features = []
    for word, freq in frequencies.items():
        # Repeat the word based on its frequency
        features.extend([word] * freq)

    return features

def get_simhash(input_text):
    # Split the input into a set of features
    features = extract_features(input_text)

    # Generate a hash for each feature
    hashes = [hashlib.sha1(feature.encode('utf-8')).hexdigest() for feature in features]

    # Combine the feature hashes to produce the final simhash
    concatenated_hash = ''.join(hashes)
    simhash = hashlib.sha1(concatenated_hash.encode('utf-8')).hexdigest()

    return simhash


def compare_simhashes(simhash1, simhash2):
    # Convert simhashes to integers
    int_simhash1 = int(simhash1, 16)
    int_simhash2 = int(simhash2, 16)

    # Calculate the distance between the simhashes
    distance = bin(int_simhash1 ^ int_simhash2).count('1')

    return distance


previous_url_hash = None

def is_same_content(text):

    global previous_url_hash
    current_hash = get_simhash(text)

    if previous_url_hash is None:
        previous_url_hash = current_hash
        return False

    distance = compare_simhashes(previous_url_hash, current_hash)

    previous_url_hash = current_hash

    if distance < 5:
        return True
    return False




















############################################################################
#                            TOKENIZATION
############################################################################

URL_LENGTH = dict()
TOKEN_COUNTS = dict()

def process_url_text(url, text):

    tokens = tokenize(text)
    _update_token_frequencies(tokens)
    URL_LENGTH[url] = len(tokens)

def tokenize(text):

    tokens = []

    for line in text:
        tokens += re.findall(r"[a-zA-Z0-9']+", line.lower())

    return tokens

def _update_token_frequencies(token_list):

    for token in token_list:
        if token not in TOKEN_COUNTS.keys():
            TOKEN_COUNTS[token] = 1
        else:
            TOKEN_COUNTS[token] += 1

def compute_word_frequencies(token_list):

    frequency_dict = dict()

    for token in token_list:
        if token not in frequency_dict.keys():
            frequency_dict[token] = 1
        else:
            frequency_dict[token] += 1

    return frequency_dict














