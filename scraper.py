import re
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
import nltk
from nltk.corpus import stopwords
import hashlib


############################################################################
#                               GLOBALS
############################################################################
VALID_DOMAINS = ("ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu")
TOTAL_URLS = set() # set of all unqiue URLs identified

ICS_SUB_DOMAIN = dict()

URL_LENGTHS = dict() # url : num_tokens
TOKEN_COUNTS = dict() #

THRESHOLD_COUNT = dict()

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}
nltk.download('stopwords')
stop_words = stopwords.words('english')

ALL_STOP_WORDS = set(stop_words).union(STOP_WORDS)

############################################################################
#                         SCRAPER MAIN FUNCTIONS
############################################################################


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

    defragmented_url = url.split('#')[0]

    if not should_extract(url, resp):  # make sure the URL is valid and is not responding with error
        return list()

    TOTAL_URLS.add(defragmented_url)  # since it's a set, no need to check if url already in or not
    is_sub_domain(url)  # count subdomain for report, above same content check

    soup = BeautifulSoup(resp.raw_response.content, 'lxml')
    text = soup.get_text()

    tokens = tokenize(text)  # get all the tokens from this URL

    if is_same_content(tokens):  # pass it to the simhash checkers
        return list()

    process_url_text(url, tokens)  # process URL tokens for report

    urls = parse_urls(resp.raw_response.url, soup) # extract all urls from the page

    valid_urls = []
    for url in urls:
        if is_valid_authority(url):  # don't even place URLs with invalid authority inside of frontier
            valid_urls.append(url)
    return valid_urls

def parse_urls(base_url, soup):
    """
    Parses and extracts absolute URLs from a BeautifulSoup object.

    Args:
        base_url (str): The base URL of the webpage.
        soup (BeautifulSoup): Parsed HTML content.

    Returns:
        list: A list of absolute URLs found in the page.
    """
    urls = []
    for link in soup.find_all("a"):
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)  # Convert to absolute URL
            defragmented_url = full_url.split('#')[0] # get rid of the fragment
            urls.append(defragmented_url)
    return urls
############################################################################
#                       URL CHECKS
############################################################################

def should_extract(url, resp):
    """
    Determines if a URL should be extracted based on various validity checks.

    Args:
        url (str): The URL being checked.
        resp: The response object containing metadata.

    Returns:
        bool: True if the URL should be extracted, False otherwise.
    """
    if resp.error:
        print('resp error')
        return False
    if not is_crawling_allowed(url):
        print('robots')
        return False
    if is_duplicate_url(url):
        return False
    if not check_file_size(resp):
        print('file size')
        return False
    if is_trap(url):
        print('trap')
        return False
    if not is_valid_authority(url):
        print('authority')
        return False
    if not is_valid(url):
        print('not valid')
        return False
    if not is_encodeable(resp):
        print('not encodeable')
        return False
    if resp.status == 404 or resp.status == 403:  # don't add to count if the URL is 404
        return False
    return True


def check_file_size(resp):
    """
    Checks if the response content size is within acceptable limits.

    Args:
        resp: The response object.

    Returns:
        bool: True if the content size is valid, False otherwise.
    """
    minimum_size = 500 # 500 bytes
    maximum_size = 35 * 1024 * 1024 # 35MB

    if not (minimum_size < len(resp.raw_response.content) < maximum_size):
        return False
    else:
        return True


def is_crawling_allowed(url, user_agent="*"):
    """
    Checks if crawling is permitted for the given URL based on robots.txt rules.

    Args:
        url (str): The URL to check.
        user_agent (str): The user agent string (default is '*', for all agents).

    Returns:
        bool: True if crawling is allowed, False otherwise.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt" # get url of robots.txt

    rp = RobotFileParser()

    try:
        rp.set_url(robots_url)
        rp.read()  # Read and parse robots.txt
        return rp.can_fetch(user_agent, url)  # Check if crawling is allowed
    except:
        return True  # Assume allowed if there is no robots.txt


def is_trap(url):
    """
    Checks if the url path contains a segment that might be a trap

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL has the keyword, False otherwise.
    """
    parsed = urlparse(url)
    path_segments = parsed.path.strip("/").split("/")
    calendar_pattern_found = any('calendar' in segment.lower() for segment in path_segments)
    event_pattern_found = any('event' in segment.lower() for segment in path_segments)
    doku_pattern_found = any('doku.php' in segment.lower() for segment in path_segments)


    if calendar_pattern_found or event_pattern_found or doku_pattern_found:
        return True

    return False


def is_encodeable(resp):
    """
    Makes sure that the content can be encoded with utf-8, resolve errors like:
    encoding error : input conversion failed due to input error, bytes 0x81 0x48 0x56 0x4B

    Args:
        resp: The response object.

    Returns:
        bool: True if the content is valid, False otherwise.
    """

    content_type = resp.raw_response.headers.get("Content-Type", "")
    if not re.match(r"text/.*", content_type) or not "utf-8" in content_type.lower():
        return False
    return True


def is_duplicate_url(url):
    """
    Checks if urls are unique

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is unique, False otherwise.
    """
    if url not in TOTAL_URLS:
        return False
    else:
        return True


def is_valid_authority(url):
    """
    Checks if url has a valid authority

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL has a valid authority, False otherwise.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    pattern = r"^(?:\w+\.)*(" + "|".join(re.escape(n) for n in VALID_DOMAINS) + r")$"

    return re.match(pattern, netloc) is not None


def is_valid(url):
    # Decide whether to crawl this url or not.
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.

    try:
        parsed = urlparse(url)
        # Check if scheme is https or http
        if parsed.scheme not in set(["http", "https"]):
            return False

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


##############################################################################################################
#                 Similarity and Trap Detection
# Simhash citations:
# https://spotintelligence.com/2023/01/02/simhash/#Example_of_SimHash_Calculation
# https://stackoverflow.com/questions/456302/how-to-determine-if-two-web-pages-are-the-same
# https://algonotes.readthedocs.io/en/latest/Simhash.html
##############################################################################################################

def extract_features(tokens):
    """
    Gets the features or values for the tokens by frequency

    Args:
        tokens (list): A list of tokens.

    Returns:
        list: The list of featured tokens
    """
    # Get word frequencies from tokens
    frequencies = compute_word_frequencies(tokens)

    # Create a feature for each word based on its frequency
    features = []
    for word, freq in frequencies.items():
        # Repeat the word based on its frequency
        features.extend([word] * freq)

    return features

def simhash(features, bit_length=64):
    """
    Computes a SimHash value for a given set of features.

    Args:
        features (list): A list of feature tokens.
        bit_length (int): The length of the hash in bits (default is 64).

    Returns:
        int: The computed SimHash value.
    """
    v = [0] * bit_length

    for feature in features:
        h = int(hashlib.sha1(feature.encode('utf-8')).hexdigest(), 16)  # Convert feature to hash
        for i in range(bit_length):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    # Convert vector to binary hash
    return sum(1 << i for i in range(bit_length) if v[i] > 0)


def hamming_distance(hash1, hash2):
    """
    Computes the Hamming distance between two SimHash values.

    Args:
        hash1 (int): First hash value.
        hash2 (int): Second hash value.

    Returns:
        int: The Hamming distance between the two hashes.
    """
    x = (hash1 ^ hash2) & ((1 << 64) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ans

# Set to store all simhashes
previous_hashes = set()


def is_same_content(tokens, threshold=3):
    """
    Checks if the hashes are similar to previously seen pages.

    Args:
        tokens (list): A list of tokenized words from the content.
        threshold (int, optional): The maximum Hamming distance allowed for content to be considered similar. Defaults to 3.

    Returns:
        bool: True if the content is similar to previous pages, False otherwise.
    """
    global previous_hashes
    current_hash = simhash(extract_features(tokens))

    if not previous_hashes:
        previous_hashes.add(current_hash)
        return False

    smallest_dist = float('inf')

    for old_hash in previous_hashes:
        dist = hamming_distance(old_hash, current_hash)
        smallest_dist = min(smallest_dist, dist)

    print(f'DISTANCE: {smallest_dist}')  # Debugging info

    if smallest_dist <= threshold:
        return True  # Similar content detected

    previous_hashes.add(current_hash)
    return False

############################################################################
#                            TOKENIZATION
############################################################################

def process_url_text(url, tokens):
    """
    Updates the token frequency count and records the token length of the given URL.

    Args:
        url (str): The URL being processed.
        tokens (list): A list of tokenized words from the URL content.
    """
    global URL_LENGTHS
    _update_token_frequencies(tokens)
    URL_LENGTHS[url] = len(tokens)

def tokenize(text):
    """
    Tokenizes the given text into words, preserving alphanumeric sequences and apostrophes.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A list of tokens extracted from the text.
    """
    tokens = []

    tokens += re.findall(r"[a-zA-Z0-9']+", text.lower())

    return tokens

def _update_token_frequencies(token_list):
    """
    Updates the global token frequency dictionary with counts of tokens in the provided list.

    Args:
        token_list (list): A list of tokens to update the frequency count.
    """
    for token in token_list:
        if token not in TOKEN_COUNTS.keys():
            TOKEN_COUNTS[token] = 1
        else:
            TOKEN_COUNTS[token] += 1

def compute_word_frequencies(token_list):
    """
    Computes the frequency of each unique token in the given list.

    Args:
        token_list (list): A list of tokens.

    Returns:
        dict: A dictionary mapping tokens to their respective frequencies.
    """
    frequency_dict = dict()

    for token in token_list:
        if token not in frequency_dict.keys():
            frequency_dict[token] = 1
        else:
            frequency_dict[token] += 1

    return frequency_dict

############################################################################
#                            REPORT GENERATION
############################################################################

def is_sub_domain(url):
    """
    Determines if a URL belongs to the ics.uci.edu subdomain and tracks occurrences.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a subdomain of ics.uci.edu, False otherwise.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    # Ensure it is either exactly "ics.uci.edu" or a subdomain (e.g., "vision.ics.uci.edu")
    if netloc == "ics.uci.edu" or netloc.endswith(".ics.uci.edu"):
        if netloc not in ICS_SUB_DOMAIN:
            ICS_SUB_DOMAIN[netloc] = 1
        else:
            ICS_SUB_DOMAIN[netloc] += 1
        return True  # It's a valid subdomain

    return False  # Not a valid subdomain



def get_top_50_tokens():
    """
    Retrieves the top 50 most frequent non-stopword tokens.

    Returns:
        dict: A dictionary of the top 50 tokens and their counts.
    """
    sorted_token_map = dict(sorted(TOKEN_COUNTS.items())) # sort by alphabetical first
    sorted_token_map = dict(sorted(sorted_token_map.items(), key=lambda item: item[1], reverse=True)) # sort by frequency
    filtered_tokens = {token: count for token, count in sorted_token_map.items() if token not in ALL_STOP_WORDS}
    top_50_tokens = dict(list(filtered_tokens.items())[:50])
    return top_50_tokens


def longest_page():
    """
    Finds the URL with the highest word count.

    Returns:
        tuple: A tuple containing the URL and the word count of the longest page.
    """
    url_dict = list(sorted(URL_LENGTHS.items(), key=lambda item: (-item[1], item[0])))

    return url_dict[0]


def print_crawler_report():
    """
    Generates a report summarizing the web crawling results, including page counts, top tokens, and subdomain data.
    The report is saved in 'results.txt'.
    """
    file = open('results.txt', 'w')
    file.write(f'Crawler Report\n\n')
    file.write("Members ID Numbers: 47403760, 35811463, 44045256, 57082516\n")
    file.write("Members Names: Peihan Cui, Keanu Jahncke, Angelina Chau, Jorge Hernan Malagon Velasquez\n\n")
    file.write(f'Total Number of Unique and valid Pages: {len(TOTAL_URLS)}\n\n')

    longest_url = longest_page()
    file.write(f'This url: {longest_url[0]} has the most words with: {longest_url[1]} words\n\n')

    top_50_tokens = get_top_50_tokens()
    file.write(f'Top 50 tokens with count:\n')
    for token, count in top_50_tokens.items():
        file.write(f'{token} -> {count}\n')
    file.write('\n')

    file.write(f'Total Number of Unique ICS Subdomains: {len(ICS_SUB_DOMAIN)}\n\n')
    output_lines = [f"https://{url}, {count}" for url, count in ICS_SUB_DOMAIN.items()]
    file.write('\n'.join(output_lines))
    file.write(f'\nEnd of Crawler Report')
    file.close()