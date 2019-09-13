# Content AI

Content AI framework/starter kit.

#### Goal
Content editors/writers are the heart of any digital news space. This starter kit API can make content publishing tools better, faster and smarter. For ex: curation, cms, api, any application backend or even FE apps.  

##### Core Benefits
* Reduces UI components.
* Saves time in writing stories.
* Helps writing better stories.
* Reduces 'time to 1st publish' duration.

### Example UI Integration
![Content AI ](static/images/content-ai.jpg "Content AI ")

#### Features
##### Predict next word based on current word.
##### Classify breaking news.
##### Classify content category or type.
##### Predict probability of high traffic content.
##### Display wikipedia summery as you type.
##### Using facial recognition detect political figures to automatically select taxonomy or content category. [Image-AI](https://github.com/nycdidar/Image-AI)
##### Find image caption by image URL. [Image-AI](https://github.com/nycdidar/Image-AI)
##### Analyze image based on several pre trained image models. [Image-AI](https://github.com/nycdidar/Image-AI)
##### Subject priority image cropping. [Image-AI](https://github.com/nycdidar/Image-AI)
##### Context specific spell checker.
##### Extract keywords & weights.

### Algorithm Used
![Recurrent Neural Network](https://qph.fs.quoracdn.net/main-qimg-6eced51767f5bcd94b32bbe50da438e9 "Recurrent Neural Network")

[Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network)

![Naive Bayes classifier](https://provalisresearch.com/uploads/linear_vs_nonlinear_problems.png "Naive Bayes classifier")

[Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

![Natural language processing](https://deeplearninganalytics.org/wp-content/uploads/2019/04/nlp.png "Natural language processing")

[Natural language processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing)

### Examples
##### Classify title
`curl http://localhost:5004/classify_title?title=Video shows jaw-dropping scale of Dorian's ferocious hit on the Bahamas`

Example Response:
`[{"category":"news","probability":"0.9941731"},{"category":"tech","probability":"0.0047676545"},{"category":"politics","probability":"0.0007970525"},{"category":"health","probability":"0.00026003388"},{"category":"business","probability":"2.2758081e-06"}]`
##### Classify breaking news
`curl http://localhost:5004/classify_breaking?title=Watch live: Latest on Hurricane Dorian as it move up U.S. coast`

Example Response:
`
[{"category":"breaking","probability":"0.7305469"},{"category":"normal","probability":"0.26945308"}]`

##### Named Entity Recognition (NER)
`curl curl http://localhost:5004/ner?content=<YOUR LARGE CONTENT BLOB>`

Example Response:
`[["London","B-LOC"],["you","O"],["Trump","B-PER"].....`

##### Predict Content Category
`curl 'http://localhost:5004/predict_vertical' --data 'content=<YOUR LARGE CONTENT BLOB>'`

Example Response:
`{"vertical_type":"Mach"}`

`{"keywords":{"politics":0.179,"obama":0.179,"election":0.179,"sanders":0.179......}}`

##### Predict Next Word
`curl 'http://localhost:5004/output?string=financial&work'`

Example Response:
`{  
   predictions:[   
         "planning"
   ]
}`

##### Context Based Spell Checker
`curl 'http://localhost:5004/output?string=financd'`

Example Response:

`{
  spell_suggestion:"finance"
}`


=============================================
##### Deploy (Using Docker Compose)
`cd docker`

`docker-compose up -d`

##### Deploy (Native)
`cd docker`

`pip install -r requirements.txt `

`cd ..`

`python server.py`

Visit http://localhost:5004/test to play around with example GUI.

##### THIS IS WORK IN PROGRESS.

#### TO DO
- Organize files/folders.
- Use more OOP structure.
- Add more comments.
- Reduce code footprints.
- and so on .. huge list.


> FULL CREDIT GOES TO EVERYONE INVOLVED IN ML/AI FIELD. THEY ARE TOTALLY RESPONSIBLE FOR MANKINDS EXTINCTION. :-)