import requests
from bs4 import BeautifulSoup
def get_news(i):
	link = 'http://www.hamshahrionline.ir/details/%s/' % i
	page = requests.get(link)
	soup = BeautifulSoup(page.content, 'html.parser')
	news = soup.find('div',class_='newsBodyCont')

	tmp = news.find('div',class_='leadContainer')
	title = news.find('h3').get_text()
	category = tmp.find('span').get_text()
	text = news.get_text().replace(category,'')
	date = soup.find('div',class_='newsToolsCont').find('span',class_='publisheDate').get_text()
	
	return title, category, text, date

def write_news(i,title, category, text, date):
	f = open('./result/%s.text' % i, 'w', encoding='utf-8')
	f.write(text)
	f.close()

	f = open('./result/%s.cat' % i, 'w', encoding='utf-8')
	f.write(category)
	f.close()

	f = open('./result/%s.title' % i, 'w', encoding='utf-8')
	f.write(title)
	f.close()

	f = open('./result/%s.date' % i, 'w', encoding='utf-8')
	f.write(date)
	f.close()

file = open('./last')
last = int(file.read())
file.close()
print("start from", last)
p = 386501
i = last

while True:
	i = (i * 383) % p
	try:
		title, category, text, date = get_news(i)
		print(i,date)
		#print(title, category, text, date)
		write_news(i, title, category, text, date)
	
		file = open('./last','w')
		file.write(str(i))
		file.close()
	except:
		print("%d failed"%i)
