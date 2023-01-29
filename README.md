# Adaptable summarization of text messages in large chat groups

Communication in chat groups with a certain goal in mind is tedious in
the private field but even more in a professional environment. According
to a McKinsey study[^1] a regular worker spends 42% of the average
workweek for communication tasks (28% reading and answering e-mails, 14%
communicating and collaborating internally) which drastically reduces
the time one is able to specifically work for the role he was hired for.
This Information overload even seems to be intensifying with the rise of
virtual and hybrid work models. In order to reduce the time needed to
follow conversations in a company group chat, this application should be
able to summarize points of view, opinions, proposals and decisions made
by multiple members during a conversation in a chat group. Google
released a \"State-of-the-Art Model for Abstractive Text
Summarization\", named Pegasus[^2] in 2020 which excels current methods
for summarizing text. Code and model checkpoints had been released on
GitHub[^3].

However, styles of language vary in different peer groups which
motivated using a Federated Learning approach for this application in
the first place. In addition to that, privacy reasons and purposes of
performance scaling further suggest such an implementation since large
companies produce lots of secret text messages from a vast amount of
users.

## Requirements
### Functional Requirements

1.  The application should be able to receive text messages from chat
    group discussions and generate a summary of the discussion.

2.  The summary should be presented to the user via the API Gateway, and
    the user should be able to provide feedback on the summary.

3.  The feedback should be processed and used to update the ML text
    summarization model.

4.  The application adapts to the style of chatting (corporate, casual,
    slang) by incorporating local training of the machine learning model
    and improving the overall model at the same time (Federated
    Learning).

5.  The average weighted central model should be sent to the Model
    registry for retrieval by new Federated Learning clients.

### Non-Functional Requirements

1.  The application should be highly available and scalable, to handle
    large volumes of input text and feedback from multiple users (big
    companies).

2.  The application should be secure, protecting user data and the model
    from unauthorized access.

3.  The application should be efficient, processing input text and
    feedback quickly and without significant delays.

4.  The application should be flexible, allowing the ML text
    summarization model to be easily updated and improved.

5.  The application should be maintainable, allowing developers to
    easily modify and troubleshoot the system as needed.


## Cloud-specific Architecture

After doing some research into how Federated Learning
(FL) could be actually implemented I decided to use Amazon Web Services
and Flower[^4] as an easy to use FL Framework (compare example code below). However, this application could be
implemented on any cloud platform, not just Amazon Web Services (AWS).

Example code for the Tokenizer and Inference service from the PegasusX documentation: [https://huggingface.co/docs/transformers/model_doc/pegasus_x](https://huggingface.co/docs/transformers/model_doc/pegasus_x)

``` python
from transformers import PegasusTokenizer, PegasusXForConditionalGeneration

model = PegasusXForConditionalGeneration.from_pretrained("google/pegasus-x-base")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-x-large")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. "
    "Nearly 800 thousand customers were scheduled to be affected by the shutoffs "
    "which were expected to last through at least midday tomorrow."
)
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")



# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False)[0]
```

The API Gateway could, for instance, be implemented using a generic web
server or API gateway framework, rather than a specific service provided
by a particular cloud vendor (Azure API Management, Google Cloud
Endpoints). The Model learning instances could as well use other managed
machine learning platforms, such as Azure Machine Learning or Google
Cloud AI Platform. The Blob Storages and Data Streams could use other
standard data storages and streaming formats, like Azure Blob Storage
and Azure Event Hubs or Google Cloud Storage and Google Cloud Pub/Sub.

However, for this project I decided to use only services from AWS.

## AWS

![Cloud architecture for a federated learning approach to adaptable chat
summarization.](https://github.com/chrisonntag/adaptable-chat-summarization/raw/main/diagrams/CloudPatterns-AWS.png)

The application is built using Amazon Web Services
(Figure [^5]), with a Virtual Private Network
connecting a user device (with the chat app on it) to the various
components of the system. The user requests a summary via an AWS API
Gateway, which is provided by a SageMaker instance that accesses a
pre-compiled model stored in a S3 Bucket (Model Artifact Bucket). The
user can then provide feedback on the summary, which is routed through
the API Gateway to another SageMaker instance for preprocessing. The
processed data is then sent via a Kinesis Data Streams instance to an
AWS Lambda Function that acts as a data collector, storing the data in a
database in another S3 Bucket.

Another SageMaker instance uses this training data to refine the ML text
summarization model on the company's private network, and stores the
updated model in the Model Artifact Bucket. Another AWS Lambda Function
serves as a Federated Learning Client, communicating with a central
Federated Learning server hosted on an EC2 instance in the AWS Public
Cloud (both based on the server-client synchronous principle used by the
Flower framework). This server initiates training rounds using Amazon
CloudWatch, and performs Federated Averaging to combine the weights
received from all FL clients into a new ML text summarization model. The
logging data from this process is sent to another Amazon CloudWatch
instance, which offers a dashboard for developers to access and verify
the number of clients and the accuracy and loss of the trained model.

Once a new model has been created, it is sent via Amazon EventBridge and
Kinesis Data Stream to an Amazon SageMaker Model registry, where it can
be retrieved by new FL clients.

## Conclusion

In conclusion, Federated Learning in particular can benefit greatly from
cloud and cloud-edge computing, as it relies on distributed, scalable,
and flexible computing resources to train and refine machine learning
models. By using cloud and cloud-edge computing, organizations can
easily adapt the summarization quality to their corporate language style
while optimizing the global model at the same time, without the need to
manage and maintain their own infrastructure.

By leveraging the vast and global network of data centers and
infrastructure provided by cloud platforms, such as Amazon Web Services,
Azure, and Google Cloud Platform, we can quickly and easily deploy and
run our applications and services without the need to manage and
maintain own infrastructure.


## References

[^1]: https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-social-economy

[^2]: https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html

[^3]: https://github.com/google-research/pegasus

[^4]: <https://flower.dev>

[^5]: ![AWS specific architecture of a federated learning approach to chat
summarization with further
annotations.](https://github.com/chrisonntag/adaptable-chat-summarization/raw/main/diagrams/CloudPatterns-AWS-single.png)
