import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Константы
BASE_URL = 'https://books.yandex.ru'
GENRES = [
    'https://books.yandex.ru/section/all/klassika-J2egRejp',
    'https://books.yandex.ru/section/all/antiutopii-ERh5RzXQ'
]
NUM_BOOKS_PER_GENRE = 10
OUTPUT_FILE = 'yandex_books_dataset.csv'


# Заголовки для запросов
def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': BASE_URL,
        'Connection': 'keep-alive',
    }


# Получение HTML
def get_html(url):
    headers = get_headers()
    response = requests.get(url, headers=headers, timeout=15)
    return response.text if response.status_code == 200 else None


# Извлечение названия книги
def get_book_title(card):
    title_elem = card.find('a', class_=lambda x: x and 'Typography_typography__c4nNy Typography_TextM16Medium__Dabd9 SnippetTitle_name__BQbi6' in x)
    return title_elem.get_text(strip=True) if title_elem else 'Нет названия'

# Извлечение URL книги
def get_book_url(card):
    title_elem = card.find('a', class_=lambda x: x and 'Link_link__O0n7o SnippetTitle_link__Rqb5o' in x)
    if title_elem and title_elem.get('href'):
        href = title_elem['href']
        return urljoin(BASE_URL, href) if href.startswith('/') else href
    return 'Нет URL'

# Извлечение автора
def get_book_author(card):
    author_elem = card.find('div', class_=lambda x: x and 'SnippetAuthorsOneLine_container__dgJrM Book_authors__2QLxD' in x)
    if author_elem:
        author_text = author_elem.get_text(strip=True)
        return clean_author_text(author_text)
    return 'Нет автора'

# Очистка текста автора
def clean_author_text(text):
    text = text.replace('и др.', '').replace('Книга', '').replace('Аудиокнига', '').strip()
    return text if text else 'Нет автора'

# Извлечение числовой оценки
def extract_numeric_value(text):
    import re
    match = re.search(r'(\d+[\s\.,]?\d*)', text)
    if match:
        num_str = match.group(1).replace(' ', '').replace(',', '').replace('.', '')
        return int(num_str) if num_str.isdigit() else None
    return None

# Извлечение оценки книги
def get_book_rating(card):
    rating_elem = card.find('div', class_=lambda x: x and 'Typography_typography__c4nNy Typography_TextS14Medium__V6tIN' in x)

    if rating_elem:
        rating_text = rating_elem.get_text(strip=True)
        import re
        rating_match = re.search(r'([\d,\.]+)', rating_text)
        if rating_match:
            rating_str = rating_match.group(1).replace(',', '.')
            try:
                return float(rating_str)
            except ValueError:
                pass
    return None

# Извлечение количества советов
def get_book_recommendations(card):
    stats_elems = card.find_all('div', class_=lambda x: x and ('Typography_typography__c4nNy Typography_TextS14Regular__8hura Emotion_count__Fu6AM' in x or 'Typography_typography__c4nNy Typography_TextS14Regular__8hura Emotion_count__Fu6AM' in x))
    for elem in stats_elems:
        text = elem.get_text(strip=True).lower()
        if 'советуют' in text:
            return extract_numeric_value(text)
    return None


# Извлечение количества оценок
def get_book_review_count(card):
    stats_elems = card.find_all('div', class_=lambda x: x and ('Typography_typography__c4nNy Typography_TextS14Regular__8hura Emotion_count__Fu6AM' in x or 'Typography_typography__c4nNy Typography_TextS14Regular__8hura Emotion_count__Fu6AM' in x))
    for elem in stats_elems:
        text = elem.get_text(strip=True).lower()
        if 'оценка' in text or 'отзыв' in text:
            return extract_numeric_value(text)
    return None


# Создание карточки книги
def create_book_card(card, genre_name):
    return {
        'название книги': get_book_title(card),
        'автор': get_book_author(card),
        'жанр': genre_name,
        'оценка': get_book_rating(card),
        'количество_оценок': get_book_review_count(card),
        'советуют': get_book_recommendations(card),
        'url': get_book_url(card)
    }


# Поиск карточек книг на странице
def find_book_cards(soup):
    cards = []

    # Способ 1: основные карточки
    cards1 = soup.find_all('div', class_=lambda x: x and ('InfinityList_container__tGTqP SectionPageContent_list__Ke8fv' in x or 'InfinityList_container__tGTqP SectionPageContent_list__Ke8fv' in x))
    cards.extend(cards1)

    return cards


# Проверка наличия следующей страницы
def has_next_page(soup, current_page):
    next_page_elem = soup.find('a', {'data-page': str(current_page + 1)})
    if not next_page_elem:
        next_page_elem = soup.find('button', {'data-page': str(current_page + 1)})
    if not next_page_elem:
        next_page_elem = soup.find('div', class_=lambda x: x and 'Pagination' in x and 'next' in x.lower())
    return next_page_elem is not None


# Получение книг с одной страницы
def get_books_from_page(html, genre_name, max_books):
    soup = BeautifulSoup(html, 'html.parser')
    cards = find_book_cards(soup)
    books = []

    for card in cards:
        if len(books) >= max_books:
            break
        books.append(create_book_card(card, genre_name))

    return books, has_next_page(soup, 1)


# Получение книг из жанра
def get_books_from_genre(genre_url, genre_name, max_books):
    books = []
    page = 1
    remaining = max_books

    while remaining > 0:
        url = genre_url if page == 1 else f"{genre_url}?page={page}"
        html = get_html(url)

        if not html:
            break

        page_books, next_page_exists = get_books_from_page(html, genre_name, remaining)
        books.extend(page_books)
        remaining -= len(page_books)

        if not next_page_exists or remaining <= 0:
            break

        page += 1
        time.sleep(random.uniform(1, 3))

    return books


# Извлечение названия жанра из URL
def extract_genre_name(genre_url):
    path = urlparse(genre_url).path
    genre_part = path.split('/')[-1]
    genre_name = genre_part.split('-')[0]
    return genre_name.replace('+', ' ').replace('_', ' ')

# Сохранение данных в CSV
def save_to_csv(books, filename):
    df = pd.DataFrame(books)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return df

# Основная функция
def main():
    all_books = []

    for genre_url in GENRES:
        genre_name = extract_genre_name(genre_url)
        genre_books = get_books_from_genre(genre_url, genre_name, NUM_BOOKS_PER_GENRE)
        all_books.extend(genre_books)
        time.sleep(random.uniform(2, 5))

    return save_to_csv(all_books, OUTPUT_FILE)


if __name__ == '__main__':
    main()