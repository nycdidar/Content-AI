import wikipedia

data_display = dict()
def classify_title(title):
    data_display['content'] = wikipedia.summary(title, sentences=5)
    data_display['category'] = wikipedia.page(title).categories
    data_display['image'] = wikipedia.page(title).images[0]
    return data_display
