import requests

from lxml import html

links = ["http://www.medetec.co.uk/slide%20scans/leg-ulcer-images/",
        "http://www.medetec.co.uk/slide%20scans/leg-ulcer-images-2/",
        "http://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-a/",
        "http://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-b/"]

data = "data/"

def download(url, destination):

    r = requests.get(url, stream=True)

    with open(destination, "wb") as f:

        for chunk in r.iter_content(chunk_size=1024):

            if chunk:

                f.write(chunk)

for link in links:

    page = requests.get(link)

    tree = html.fromstring(page.content)

    images_urls = tree.xpath("//a//img/@src")

    for url in images_urls:

        name = url.split('/')[1]

        complete = link + 'images/' + name

        print('Downloading ' + name + ' ...')

        download(complete, data + name)
