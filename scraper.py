import re
from collections import Counter
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
TOTAL_URLS = set() # set of all unqiue URLs identified, might need to change to a dictionary to get urls from each page

ICS_SUB_DOMAIN = dict()

URL_LENGTHS = dict() # url : num_tokens
TOKEN_COUNTS = dict() #

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
    global COUNT
    print(COUNT)
    COUNT += 1


    if not should_extract(url, resp):  # make sure the URL is valid and is not responding with error
        return list()


    soup = BeautifulSoup(resp.raw_response.content, 'html.parser')
    text = soup.get_text()

    tokens = tokenize(text) # get all the tokens from this URL

    is_sub_domain(url)  # count subdomain for report, above same content check

    if is_same_content(tokens): # pass it to the simhash checkers
        print(f'SKIPPED {url}')
        return list()

    process_url_text(url, tokens) # process URL tokens for report

    urls = parse_urls(resp.raw_response.url, soup)
    return urls


def parse_urls(base_url, soup):
    urls = []
    for link in soup.find_all("a"):
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)  # Convert to absolute URL
            defragmented_url = full_url.split('#')[0]
            urls.append(defragmented_url)
    return urls
############################################################################
#                       URL CHECKS
############################################################################


def should_extract(url, resp):
    if resp.error:
        return False
    if not is_crawling_allowed(url):
        return False
    if not check_file_size(url, resp):
        return False
    if is_calendar_trap(url):
        return False
    if not is_valid_authority(url):
        return False
    if not is_valid(url):
        return False
    if is_duplicate_url(url):
        return False

    return True


def check_file_size(url, resp):

    minimum_size = 500 #500 bytes
    maximum_size = 35 * 1024 * 1024 #35MB


    if not (minimum_size < len(resp.raw_response.content) < maximum_size):
        print(f'{url}: SIZE INVALID: {len(resp.raw_response.content)} bytes')
        return False
    else:
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


def is_calendar_trap(url):
    parsed = urlparse(url)
    path_segments = parsed.path.strip("/").split("/")
    calendar_pattern_found = any('calendar' in segment.lower() for segment in path_segments)
    event_pattern_found = any('event' in segment.lower() for segment in path_segments)

    if calendar_pattern_found or event_pattern_found:
        return True

    return False


def is_duplicate_url(url):
    # checks if urls are unique
    if url not in TOTAL_URLS:
        TOTAL_URLS.add(url)
        return False
    else:
        return True


# def is_recursive_url(url, threshold=3):
#     # Detects recursive traps where path segments are repeated too many times.
#     parsed = urlparse(url)
#     path_segments = parsed.path.strip("/").split("/")  # Extract path segments
#
#     # Count occurrences of each segment
#     segment_counts = Counter(path_segments)
#
#     # If any segment appears more than the threshold, it's likely a trap
#     for segment, count in segment_counts.items():
#         if count >= threshold:
#             return True  # Recursive pattern detected
#
#     return False  # No recursive pattern


def is_valid_authority(url):
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

    try:
        parsed = urlparse(url)
        # Check if scheme is https or http
        if parsed.scheme not in set(["http", "https"]):
            return False

        # # Check if the path of the URL is recursive
        # if is_recursive_url(url):
        #     #print('recursive', url)
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

def extract_features(tokens):
    # Get word frequencies from tokens
    frequencies = compute_word_frequencies(tokens)

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

def is_same_content(tokens):

    global previous_url_hash
    current_hash = get_simhash(tokens)

    if previous_url_hash is None:
        previous_url_hash = current_hash
        return False

    distance = compare_simhashes(previous_url_hash, current_hash)

    previous_url_hash = current_hash

    print('previous hash: ', previous_url_hash, "current_hash: ", current_hash, "distance: ", distance)
    if distance < 5:
        return True
    return False

############################################################################
#                            TOKENIZATION
############################################################################

def process_url_text(url, tokens):
    global url_lengths
    _update_token_frequencies(tokens)
    url_lengths[url] = len(tokens)

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

############################################################################
#                            REPORT GENERATION
############################################################################
def is_sub_domain(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.endswith("ics.uci.edu"):
        if netloc in ICS_SUB_DOMAIN:
            ICS_SUB_DOMAIN[netloc] = 1
        else:
            ICS_SUB_DOMAIN[netloc] += 1


def get_top_50_tokens():
    sorted_token_map = dict(sorted(TOKEN_COUNTS.items()))
    sorted_token_map = dict(sorted(sorted_token_map.items(), key=lambda item: item[1], reverse=True))
    filtered_tokens = {token: count for token, count in sorted_token_map.items() if token not in ALL_STOP_WORDS}
    top_50_tokens = dict(list(filtered_tokens.items())[:50])
    return top_50_tokens


def longest_page():

    url_dict = list(sorted(URL_LENGTHS.items(), key=lambda item: (-item[1], item[0])))

    return url_dict[0]


def print_crawler_report():
    file = open('results.txt', 'w')
    file.write(f'\t\t\t\t\tCrawler Report\t\t\t\t\t\n')
    file.write("Members ID Numbers: 47403760, 35811463, 44045256, 57082516\n\n")
    file.write(f'\t\t\tTotal Number of Unique Pages: {len(TOTAL_URLS)}\n\n')

    longest_url = longest_page()
    file.write(f'\t\t\tThis url: {longest_url[0]} has the most words with: {longest_url[1]} words\n\n')

    top_50_tokens = get_top_50_tokens()
    for token, count in top_50_tokens.items():
        file.write(f'\t\t\t{token} -> {count}\n')
    file.write('\n')

    file.write(f'\t\t\tTotal Number of Unique ICS Subdomains: {len(ICS_SUB_DOMAIN)}\n\n')
    output_lines = [f"\t\t\thttps://{url}, {count}" for url, count in ICS_SUB_DOMAIN]
    file.write('\n'.join(output_lines))
    file.write(f'\t\t\t\t\tEnd Crawler Report\t\t\t\t\t\n')
    file.close()