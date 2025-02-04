import re
import sys


def tokenize(text_file_path):
    """
    Runtime Complexity: O(n)

    The function loops through all lines of the file one by one and process them
    so if n is the number of characters, that should be O(n) as re.split will look through all characters.
    Then the function makes sure all elements in result is not an empty string so that will also take O(n) worse case.
    Since asymptotic notation ignores lesser expressions, it would just be O(n).

    Reads in a text file and returns a list of tokens in that file.
    A token is a sequence of alphanumeric characters, independent of capitalization

    :param text_file_path: Path to the text file to be read
    :return: list of tokens in that file
    """

    result = []
    with open(text_file_path) as f:
        for index, line in enumerate(f):
            split_line = re.split('[^a-zA-Z0-9]', line)
            result += ([x.lower() for x in split_line])
    result = [x for x in result if x]  # get rid of all empty strings from re.split
    return result


def compute_word_frequencies(token_list):
    """
    Runtime Complexity: O(n)

    This function also just takes O(n) where n is the number of tokens in the token list
    since it is looping through the token list and putting everything in the token map

    Counts the number of occurrences of each token in the token_list.

    :param token_list: List of tokens
    :return dict, mapping each token to the number of occurrences
    """

    # put everything in a map, if not already in it, then add it, if it is in it, add 1 to the value
    token_map = {}
    for x in token_list:
        if x != '':  # double checks to make sure no empty strings
            if x in token_map:
                token_map[x] = token_map[x] + 1
            else:
                token_map[x] = 1
    return token_map


def print_frequencies(token_map):
    """
    Runtime Complexity: O(nlogn)

    This function loops through the token map and prints each token with
    its frequency. That will take O(n) time, where n is the number of tokens.
    Since the sorted function takes O(nlogn) time, that means it dominates the runtime so
    print_frequencies takes O(nlogn)

    Prints out the word frequency count, ordered by decreasing frequency.

    :param token_map: token_list: dict, mapping each token to the number of occurrences
    """
    sorted_token_map = dict(sorted(token_map.items()))
    sorted_token_map = dict(sorted(sorted_token_map.items(), key=lambda item: item[1], reverse=True))

    for token, freq in sorted_token_map.items():
        print(token, '=', freq)


def main():
    """
    Runtime Complexity: O(nlogn)

    Just calls all the functions, the one that takes the longest is O(nlogn), which means that
    dominates the runtime, so the runtime of PartA as a whole, is O(nlogn)
    """
    try:
        if len(sys.argv) != 2:
            print("Error: Invalid number of arguments!")
        else:
            tokens = tokenize(sys.argv[1])
            frequency = compute_word_frequencies(tokens)
            print_frequencies(frequency)
    except FileNotFoundError:
        print('File was not found')
    except PermissionError:
        print('Cannot access file')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
