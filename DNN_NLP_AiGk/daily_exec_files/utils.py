from bs4 import BeautifulSoup
import requests
import logging
import pymongo
import multiprocessing
import dateparser
import pandas as pd
import datetime
import numpy as np
import re
from tqdm import tnrange,tqdm_notebook


from bs4 import BeautifulSoup
#import requests
import logging
import pymongo
import multiprocessing
import dateparser

def _new_blog_content(url): #Scrape the main page of mustread articles of forum_ias
    html_doc = requests.get(url, timeout=180).content #requests is a modelue for getting permissions of a HTML page
    soup = BeautifulSoup(html_doc, 'html.parser') #html_doc contains all the content in the web page
    content = soup.find_all('div', {'class': 'pf-content'})[-1] #get all the contents of the specified class
    return content


def _get_mustread_urls(index_url='https://blog.forumias.com/page/mustread'): #from this page extract all the urls based on date and also archives
    content = _new_blog_content(index_url)
    mustread_links = [] #save the urls in a list
    for a in content.find_all('a'):
        try:
            if 'must read' in a['title'].lower():
                mustread_links.append(a['href']) #a['href'] generally contians url link
        except KeyError:
            pass

    return mustread_links


def get_hindu_uid(url): #obtain hindu UID which is article's unique id
    return 'hindu_' + url.split('/')[-1].replace('.ece', '')


def _scrape_mustread_url(mustread_url): #each must read article has many other links from sources like the hindi or indian express.Take important features from them
    content = _new_blog_content(mustread_url)
    df  = pd.DataFrame()
    url = []
    hindu_uid = []
    forum_ias_title = []
    for a in content.find_all('a'):
        if 'thehindu' in a['href']:
            url.append(a['href']) #append links to URL list
            hindu_uid.append(get_hindu_uid(a['href']))
            forum_ias_title.append(a.get_text().strip())
    df['url'] = url
    df['hindu_uid'] = hindu_uid
    df['forum_ias_title'] = forum_ias_title
    return df


Logger = []
def _scrape_err_catch(mustread_url): #error handling with try catch
    try:
        return(_scrape_mustread_url(mustread_url))
    except (IndexError, requests.exceptions.ConnectionError,
            pymongo.errors.DuplicateKeyError):
        Logger.append(mustread_url)
        return(pd.DataFrame([]))
    
    
    
def _get_gk_id(url): #extract unique id of article from GK today link
    return 'gktoday_' + url.split('-')[-1].replace('.html', '')


#gktoday will have two articles in a single page.
def scrape_page_gktoday(URL): 
    html_doc = requests.get(URL).content
    soup = BeautifulSoup(html_doc, 'html.parser')
    page = soup.find('div', {'class': "posts-listing"}) #Main page has articles in posts-listing class in div element
    articles = soup.find_all('div', 'post-content') #div:post_content are the actual articles needs to be scraped

    url = []
    title_ = []
    date_ = []
    id_ =[]
    html = []
    text = []
    all_article_jsons = []
    df = pd.DataFrame()
    for article in articles:
        title = article.find('h1') #title is present in h1 element 
        link = title.find('a') #link in a
        date = article.find('div', 'postmeta-primary') #extract date in postmeta-primary
         #append the extracted values to the lists created above
        url.append(link['href'])  
        title_.append(link.get_text())
        date_.append(dateparser.parse(date.get_text()))
        id_.append(_get_gk_id(link['href']))
           # remove non article elements
        title.decompose()
        date.decompose()
        try:
            article.find('div', 'featured_image').decompose()
        except AttributeError:
            pass

        html.append(article.prettify()) #will enable us to view how the tags are nested in the document
        text.append(article.get_text()) #extract text
        
    df['url'] = url
    df['title'] = title_
    df['date'] = date_
    df['gk_today_uid'] = id_
    df['html'] = url
    df['text'] = text

#As the have two articles in the same page,tags and categories are extarcted separately
    tags_categories = soup.find_all('div', {'class': 'small-font'}) #extracting tags and categories in small-font class
	#extract categories that have 'category tag' in rel element
    categories = []
    categories_ = []
    for i in tags_categories[0].find_all('a', attrs={"rel": "category tag"}): #for 1st article
        categories_.append(i.get_text().strip())
    categories.append(categories_)
    categories_ = []  
    for i in tags_categories[1].find_all('a', attrs={"rel": "category tag"}):#for 2nd article
        categories_.append(i.get_text().strip())

    categories.append(categories_)


    tags_categories = soup.find_all('div', {'class': 'small-font'})
	#extract tags that have 'tag' in rel element
    tags = []
    tags_ = []
    for i in tags_categories[0].find_all('a', attrs={"rel": "tag"}): #for 1st article
        tags_.append(i.get_text().strip())
    tags.append(tags_)
    tags_ = []  
    for i in tags_categories[1].find_all('a', attrs={"rel": "tag"}):#for 2nd article
        tags_.append(i.get_text().strip())
    tags.append(tags_)

    df['categories'] = categories
    df['tags'] = tags
    return(df)
    
    

        
        
Logger = []
def _scrape_err_catch_gktoday(page_url): #error handling using try except
    try:
        return(scrape_page_gktoday(page_url))
    except (IndexError, requests.exceptions.ConnectionError,
            pymongo.errors.DuplicateKeyError):
        Logger.append(page_url)
        return(pd.DataFrame([]))
    
    
def scrape_archive_url_hindu(archive_url):
    all_articles = []
    archive_date = archive_url[-11:-1] #we have url with date as suffix where we scrape the urls of those particular dates from hindu (archive [-11:-1] gives the date) 
    html_doc = requests.get(archive_url, timeout=30).content
    soup = BeautifulSoup(html_doc, 'html.parser')
    tpaper_container = soup.find('div', {'class': 'tpaper-container'})
    sections = tpaper_container.find_all('section') 
    #update title url uid heading date into list and then to a dataframe
    title = []
    url = []
    uid = []
    section_heading = []
    df = pd.DataFrame()
    for section in sections: 
        articles_list = section.find('ul', {'class': 'archive-list'}).find_all('a')
        for article in articles_list: #'a' has the details of heading/title,url,uid in href element
            title.append(article.text)
            url.append(article.get('href'))
            uid.append(get_hindu_uid(article.get('href')))
            section_heading.append(section.find('h2', {'class': 'section-heading'}).get_text().strip()) #section heading
            date_ = archive_date
    df['title'] = title
    df['url'] = url 
    df['uid'] = uid 
    df['section_heading'] = section_heading 
    df['date_'] = date_ 
    return(df) #store keys in df 

def get_hindu_uid(url):
    url = url.split('?')[0]
    return 'hindu_' + url.split('/')[-1].replace('.ece', '')




        
def _scrape_err_catch_hindu(archive_url):
    try:
        return(scrape_archive_url_hindu(archive_url))
    except (IndexError, requests.exceptions.ConnectionError,
            pymongo.errors.DuplicateKeyError):
        Logger.append(archive_url)
        return(pd.DataFrame([]))
    
    
    
def get_businessline_uid(url):
    return 'businessline_' + url.split('/')[-1].replace('.ece', '')



def scrape_archive_url_businessline(URL):
    html_doc = requests.get(URL).content
    soup = BeautifulSoup(html_doc, 'html.parser')
    content = []
    for i in soup.find_all('section',{'class': 'paddingLF30'}): #all the content we need to extract is in this class
        soup2 = i.find_all('a') #'a' has links to few articles with particular category like state,commodities etc.
        for j in soup2:
            if j.get('href')!=None: #href has links to the articles during that particluar date and title of article
                content.append([j.text,j.get('href'),soup2[0].text,get_businessline_uid(j.get('href'))])
            
    df = pd.DataFrame()
    content = np.array(content) #content above is a list not we convert to array
    df['title'] = content.T[0] #transpose and load the attributes into a df
    df['links'] = content.T[1]
    df['category'] = content.T[2]
    df['uid'] = content.T[3]
    df['date'] = URL.split('web/')[1][:-1] #date is already in url so spilt after web/ and load it
    df = df.dropna()
    return(df)    


def _scrape_err_catch_businessline(archive_url_businessline): #called in businessline.py where url is given with date
    try:
        return(scrape_archive_url_businessline(archive_url_businessline))
    except (IndexError, requests.exceptions.ConnectionError,
            pymongo.errors.DuplicateKeyError):
        Logger.append(archive_url_businessline)
        return(pd.DataFrame([]))



    
    
def filter_tags(soup, filters):
    for f in filters:
        for x in soup.find_all(*f):
            x.decompose()
    return soup

def scrape_hindu_article(url):
    r = requests.get(url)
    r.raise_for_status()
    html_doc = r.content

    soup = BeautifulSoup(html_doc, 'html.parser')
    main_article = soup.find('div', {'class': 'article'})

    main_article = filter_tags(main_article, [  #why are we doing it like this?
        ['script'],
        ['style'],
        ['div', {'class': 'support-jlm'}],
        ['div', {'class': 'artcl-social-media'}],
        ['div', {'class': 'ad-container'}],
        ['div', {'class': 'article-exclusive'}],
        ['div', {'class': 'mobile-author-cont'}],
        ['span', {'class': 'more-in'}]
    ])
    document = {}

    footer = [x for x in main_article.children if x.name == 'div'][-1]  
    tags_container = footer.find('div', {'class': 'morein-tag-cont'}) #tags are in footer
    tags = [x.get_text().strip() for x in
            tags_container.find_all('a', {'class': 'txt'})]

    try:
        section = tags_container.find(
            'a', {'class': 'section-button'}).get_text().strip()
        document['section'] = section
    except AttributeError:
        pass
    footer.decompose()

    try:
        title = main_article.find('h1', {'class': 'title'})
        document['title'] = title.get_text().strip()
        title.decompose()
    except:
        pass

    author_container = main_article.find('div', {'class': 'author-container'})
    author = author_container.find('span', {'class': 'author-img-name'})
    try:
        author = author.get_text().strip()
    except AttributeError:
        author = None

    location_date = author_container.find_all(
        'span', {'class': 'ksl-time-stamp'})[: 2]

    if len(location_date) == 2:
        location, date = location_date
        location = location.get_text().strip()
    else:
        location = None
        date = location_date[0]

    author_container.decompose()
    text = []
    text = main_article.get_text().strip() #extract text
    id_ = get_hindu_uid(url)
    return [tags,author,location,date,text,main_article]

#url = 'https://www.thehindu.com/news/national/statue-of-unity-sholisted-for-uk-based-structural-award/article28818509.ece'
#tags,author,location,date,text,html = scrape_hindu_article(url)


def scrape_businessline_article(url):
    html_doc = requests.get(url).content
    soup = BeautifulSoup(html_doc, 'html.parser')
    main_article = soup.find('div', {'class': 'contentbody'}) #get the content of the webpage here (text,heading,date,image etc.,)
    
    #we have two divs where we have date in the web page some pages have either one or two
    if main_article.find('div',{'class':"publisheddate marginBottom10 clearboth"}) !=None: #check in this div element and extract date
        date = main_article.find('div',{'class':"publisheddate marginBottom10 clearboth"}).span.none.text.replace('\n','')
    else: 
        date = main_article.find('div',{'class':"inf-scroll-pubdate paddingTopBottom20"}).span.none.text.replace('\n','')
    #extract tags from class tag-button
    topics = []
    tags = main_article.find_all('div',{'class':'tag-button'}) 
    for i in range(len(tags)):
        topics.append(tags[i].text.replace('\n','')) #adding tags

    headings = []
    h = main_article.find_all(re.compile(r'h\d+'))#**complie(r'h\d+') ??
    for i in range(len(h)):
        headings.append(h[i].text.replace('\n',''))

    decomp = [main_article.find_all('img'),
     main_article.find_all('script'),
     main_article.find_all('br'),
     main_article.find_all('div',{'class':'business-disclaimer'}),
     main_article.find_all('div',{'class':'tag-button'}),
     main_article.find_all('div',{'class':'clear hidden-xs hidden-sm text-center'}),
     main_article.find_all('div',{'class':'publisheddate marginBottom10 clearboth'}),            
     main_article.find_all('div',{'class':'relatedtag'}),
     main_article.find_all(re.compile(r'h\d+'))]
    for i in decomp:
        for j in i:
            j.decompose()  #not sure about deompse func
    content = [] #extract text
    curated_article = main_article.find_all('p') #can also use get_text.strip() like in hindu
    for i in curated_article:
        content.append(i.text.replace('\n','')) #append all paragraph tests to content list
    return(date,topics,headings,content,main_article)


#url = 'https://www.thehindubusinessline.com/companies/paytm-eyeing-20-crore-accounts-in-first-year/article8693678.ece'
#date,tags,headings,text,html = scrape_businessline_article(url)