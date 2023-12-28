from itertools import product
from concurrent.futures import ThreadPoolExecutor

from data_preparation.data_parameters import VENUES, LOCATIONS, N_PAGES
from data_preparation.scrape_reviews import scrape_reviews

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=4)
    executor.map(lambda venue, location: scrape_reviews(venue, location, N_PAGES),
                 *zip(*list(product(VENUES, LOCATIONS))))
