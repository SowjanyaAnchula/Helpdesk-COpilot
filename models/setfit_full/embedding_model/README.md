---
base_model: sentence-transformers/all-MiniLM-L6-v2
library_name: setfit
metrics:
- accuracy
pipeline_tag: text-classification
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'SaaS Platform Experienced Unforeseen Crash Following PostgreSQL Update


    Hello Customer Support, I am contacting you to report an issue we are facing with
    our SaaS platform. It crashed unexpectedly today, and we believe it might be because
    of a recent PostgreSQL update. We have attempted to restart the server and reviewed
    the logs, but we have not yet managed to resolve the problem. The crash happened
    with no prior warning, and we are worried about its potential impact on our users.
    Could you please help us address this issue promptly? We would greatly appreciate
    any advice or assistance you can offer to get our platform back online.'
- text: 'Poor Performance


    Experienced sluggish performance in the project management SaaS application amid
    heightened user activity and limited resources.'
- text: 'Technical Problem with Investment Optimization Tool


    The investment optimization tool has experienced a crash during data processing,
    which we suspect is due to server overload. Attempts to resolve the issue by rebooting
    the TP-Link switch and restarting Apache Hadoop have not been successful. The
    tool continues to crash while processing a large dataset, even after restarting
    services. We would appreciate your prompt assistance in resolving this issue.'
- text: 'Problem with Application Crashing


    Facing difficulties with application crashing'
- text: 'Support Recent Security Incident Report


    Dear Customer Support, I am writing to report a recent security incident that
    occurred in our hospital. There was an unauthorized access incident that compromised
    medical data on our hospital systems. The incident might have resulted from outdated
    security protocols or user errors. We have already tried updating our security
    software and reviewing access logs, but the issue persists. We would greatly appreciate
    your guidance on how to resolve the issue and prevent future incidents. Please
    let us know the next steps. Thank you for your assistance.'
inference: true
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 10 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label                           | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Technical Support               | <ul><li>"Reporting Security Incident for Customer Support\n\nCustomer Support, we are reporting a security incident where potentially sensitive medical data within the hospital's systems may have been exposed. The issue might be due to outdated software vulnerabilities or misconfigured settings. We have attempted to reset the security protocols and applied available patches, but the problem still persists. We kindly request your assistance in resolving this matter as soon as possible to ensure the integrity of the systems. Please let us know the next steps you would like us to take. Thank you."</li><li>'Assistance with Hadoop Support\n\nCould you offer guidance on optimizing the integration of Apache Hadoop 3.2.1 within a scalable SaaS project management platform? I would appreciate any tips or best practices.'</li><li>'Severe Performance Degradation Noted During Peak User Activity\n\nSignificant performance decline witnessed during the peak usage hours. This might be attributed to a surge in user activity and inefficient database queries. Efforts to optimize the database queries have resolved the issue.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Customer Service                | <ul><li>'Inquiry About Vagrant Integration With SaaS Platform\n\nHello Customer Support, I am contacting you to ask about the Vagrant integration capabilities for your project management SaaS platform. I am keen to discover how Vagrant can be utilized to boost our development processes and foster better teamwork. Could you please elaborate on the tools and features that are supported? Additionally, I would be grateful if you could point me towards any instructional materials or documentation that could assist me in starting the process. Thank you for your support, and I am eagerly awaiting your response.'</li><li>'The SaaS platform faced sporadic outages caused by compatibility problems with Node.js 14.17 during Ansible deployment processes. Restoring previous Node.js versions and restarting services successfully resolved the problem.'</li><li>'request for improvements in security protocols for medical data management systems. this involves updates to both software and hardware to guarantee the confidentiality and integrity of patient information. consider these updates crucial for protecting sensitive data and maintaining trust in the systems. would greatly appreciate if you could review the matter and provide a plan for implementation.'</li></ul>                                                                                                                                                                                                                                                                                                                                                               |
| Sales and Pre-Sales             | <ul><li>'Customer support team, I am reaching out to inquire about how data analytics services can influence investment optimization strategies. Could you please offer detailed insights into how this service enhances strategic decision-making? I would also appreciate information about the benefits and features of the service to better understand how it can support investment choices. Thank you for your assistance.'</li><li>'Inquiry About Digital Strategies Solutions\n\nCustomer Support is reaching out to inquire about Digital Strategies services aimed at enhancing brand development through the offered service packages. Could you please share details of the various options available and how they could benefit our business? Additionally, I would appreciate information regarding the pricing included in each package. Thank you, and I look forward to your prompt response.'</li><li>'Investment Prediction Inaccuracy Inquiry\n\nDear Customer Support, I am encountering inaccurate investment forecasts. This could be because of the insufficient data analysis tools available. Despite updating the software and retraining the models, the problem still exists. Please investigate and suggest a solution. Kindly request any further information you might need from me to address this issue. Thank you for your time and support. I am looking forward to your prompt response.'</li></ul>                                                                                                                                                                                                                                          |
| Product Support                 | <ul><li>'Multiple Tools Malfunctioning Post Update\n\nHello customer support, several tools have stopped functioning at the same time. It is possible that a recent update may have introduced compatibility problems. Despite restarting all the affected devices, the issue still continues. Can you assist me in resolving this matter urgently? I have attempted troubleshooting steps, but to no avail. The tools were operating normally before the update, but now they are not working. Any help you can provide to restore their functionality would be greatly appreciated. Thank you for your attention and support.'</li><li>'Please Improve Compatibility for SaaS\n\nRequest to make the project management SaaS compatible with PowerPoint 2021 and Excel to enhance user experience and streamline workflow processes.'</li><li>'Guidance on Using Ring Light with Tripod Stand\n\nHello Customer Support, I am contacting you with a request for information on the training resources available for enhancing the use of my newly purchased Ring Light with Tripod Stand. I am enthusiastic about learning how to utilize this item to its fullest extent. Could you kindly share any accessible training materials, such as instructional videos, workshops, or online courses? Besides, I would be grateful for any advice on the optimal setup and handling of the product. I eagerly await your response and am looking forward to your prompt reply. Thank you for your support.'</li></ul>                                                                                                                                                                 |
| Returns and Exchanges           | <ul><li>"Incomplete Product Bundle Received From Your Company\n\nDear customer support, I recently found that my product bundle, which was supposed to include several items, was incomplete upon opening. Despite attempts to contact your team through phone and email, I haven't received any responses. I suspect this might be due to an error in your order fulfillment system. Could you please look into this and provide a solution? If you need more information from me, please let me know."</li><li>'Required Assistance with Investment Predictions\n\nRecently, investment forecasts did not align with actual market movements, which might be due to using obsolete data models. Efforts have been made to update the analytical tools, but the issue remains unresolved.'</li><li>'Query on Product Integration Features\n\nHello Customer Support, I am inquiring about the integration capabilities of our product with Salesforce CRM and Smartsheet. Could you provide detailed information on how the product integrates these tools and what types of data are synced? I would also appreciate any information on potential limitations. Looking forward to hearing back from you. I am excited to learn how this product can help our business. Please let me know if you need any additional information from me.'</li></ul>                                                                                                                                                                                                                                                                                                                             |
| Billing and Payments            | <ul><li>'Drupal Commerce Assistance\n\nCustomer support inquiry regarding detailed billing and payment options for integrating the Drupal Commerce SaaS platform. Could you please share information about available payment gateways, subscription plans, and any additional fees related to the integration? I would also appreciate it if you could provide documentation or guides to assist with the setup. Thank you for your time and assistance. I look forward to your response soon.'</li><li>'Support for Norton 360\n\nInquiring about Norton 360 support to enhance digital marketing security strategies as a business owner. Looking to protect the online presence and ensure the security of customer data. Would appreciate guidance on how to effectively utilize Norton 360 to achieve these goals. Specifically, interested in learning about the features and tools available in Norton 360 to strengthen digital marketing security, including anti-virus and anti-malware protection.'</li><li>'Assistance with Securing Medical Data\n\nCustomer Support, I am requesting assistance in implementing enhanced security measures to better protect the medical data within our hospital infrastructure. Due to the increasing threat of cyberattacks, it is crucial to take proactive measures to safeguard sensitive patient information and prevent potential breaches. I would appreciate guidance on best practices for securing medical data, along with recommendations for robust security solutions that can be integrated with our existing systems. Additionally, I am interested in learning about the latest security developments.'</li></ul> |
| IT Support                      | <ul><li>'Problem with File Uploads\n\nThe system is crashing during project file uploads across multiple devices, which might be related to compatibility issues with Cassandra 4.0.'</li><li>'Enhancing Digital Brand Expansion\n\nWould you like details on digital strategies for brand growth services?'</li><li>'Combining Marketing Processes\n\nDear Customer Support Team,\\n\\nI hope this message reaches you well. I am currently engaged in enhancing the efficiency of marketing activities across multiple departments, including creative, analytics, and content teams. Recently, we have incorporated new storage hardware along with connected smart infrastructure. Our goal is to ensure smooth management and optimal utilization of these resources, and we are seeking expert advice.\\n\\nSpecifically, I would like to learn about best practices for integrating various marketing workflows with the newly implemented hardware storage solutions.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Service Outages and Maintenance | <ul><li>'Urgent Problem\n\nService interruptions impacting various devices and essential software vital for our data analysis operations. These outages are causing substantial disruptions and must be resolved promptly to prevent further losses.'</li><li>'Advice on Safeguarding Medical Data with Drupal Commerce and Salesforce CRM\n\nI am reaching out to seek guidance on securing medical data using Drupal Commerce and Salesforce CRM. Given the highly sensitive nature of medical data, it is crucial to handle, store, and integrate these systems with utmost care to ensure their security and integrity. Could you provide recommendations or resources that can help achieve this? Your guidance on the matter would be greatly appreciated. Thank you.'</li><li>'Problem with Service\n\nA significant service outage has impacted multiple devices running software that utilizes data analytics. The issue might have arisen from a conflict during a software update or a power failure. Despite rebooting the affected systems and reviewing the configuration settings, we have not resolved the problem. The outage is currently hindering our ability to process and analyze data, leading to substantial delays. We urgently request your assistance in resolving this matter, as it is critical for our operations. Please respond at your earliest convenience.'</li></ul>                                                                                                                                                                                                                                                                          |
| Human Resources                 | <ul><li>'Support Inquiry for Reported System Outage\n\nCustomer Support, we are reporting an issue with the report system outage that affected access to our project management tools. The problem may have arisen from a recent software update. Despite attempts to restart affected applications and devices, the issue continues. We would greatly appreciate if you could look into this matter and provide a solution as soon as possible. Please let us know if there is any additional information needed to facilitate the resolution of this issue. Thank you for your time and assistance with this matter.'</li><li>"Detection of Unauthorized Access Attempt\n\nAn unauthorized access attempt was detected on the hospital's network. This may have occurred due to outdated security protocols. We have updated access credentials and conducted a basic security audit. The issue has been thoroughly investigated to prevent future occurrences. Our team is working diligently to ensure the security of the network. Please stay updated on the developments. If you have any questions or concerns regarding this matter, please let us know."</li><li>"Concern Regarding Data Breach in Hospital Systems\n\nA data breach has been detected in the hospital's systems, which could potentially compromise medical records. This issue might be due to outdated security protocols and inadequate employee oversight. It is recommended to update the software, enhance staff training, and strengthen data protection measures to ensure secure data."</li></ul>                                                                                              |
| General Inquiry                 | <ul><li>'Enquiry on Security Protocols for Protecting Medical Data in Hospital IT Systems\n\nI am writing to seek information on the security measures suggested for safeguarding medical data in hospital IT networks. Given the rising threat of cyber attacks and data leaks, it is vital to secure sensitive patient information. I would greatly appreciate any advice or materials your team can offer. Kindly inform me if there are particular protocols or steps that you recommend for securing medical data.'</li><li>'Details on Integrating AutoCAD 2022 SaaS\n\nSeeking information on the integration of AutoCAD 2022 SaaS for project management. Would appreciate it if you could outline the benefits this integration would offer our team. Currently, we use AutoCAD for design and SaaS for project management, and we are looking to streamline our workflow by integrating the two systems. Please inform me of the necessary steps we need to take to make this happen. I am looking forward to your response.'</li><li>'Report on Inaccurate Investment Forecasts\n\nGreetings, I am contacting you to report an issue with the investment forecasts which have shown inaccurate projections. There is a possibility that this could be due to a data integration issue. I have already checked the data sources and referred to the user manual, yet the problem still remains unresolved. I would greatly appreciate if you could investigate this and provide either a solution or instructions on how to rectify it. Please inform me if you require any additional information from me.'</li></ul>                                                   |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Problem with Application Crashing

Facing difficulties with application crashing")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 5   | 60.8188 | 155 |

| Label                           | Training Sample Count |
|:--------------------------------|:----------------------|
| Billing and Payments            | 32                    |
| Customer Service                | 32                    |
| General Inquiry                 | 32                    |
| Human Resources                 | 32                    |
| IT Support                      | 32                    |
| Product Support                 | 32                    |
| Returns and Exchanges           | 32                    |
| Sales and Pre-Sales             | 32                    |
| Service Outages and Maintenance | 32                    |
| Technical Support               | 32                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (3, 3)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0013 | 1    | 0.3211        | -               |
| 0.0625 | 50   | 0.2036        | -               |
| 0.125  | 100  | 0.2153        | -               |
| 0.1875 | 150  | 0.2326        | -               |
| 0.25   | 200  | 0.2486        | -               |
| 0.3125 | 250  | 0.2123        | -               |
| 0.375  | 300  | 0.1436        | -               |
| 0.4375 | 350  | 0.2078        | -               |
| 0.5    | 400  | 0.2292        | -               |
| 0.5625 | 450  | 0.2478        | -               |
| 0.625  | 500  | 0.1722        | -               |
| 0.6875 | 550  | 0.1602        | -               |
| 0.75   | 600  | 0.2385        | -               |
| 0.8125 | 650  | 0.1862        | -               |
| 0.875  | 700  | 0.2194        | -               |
| 0.9375 | 750  | 0.1328        | -               |
| 1.0    | 800  | 0.2226        | -               |
| 1.0625 | 850  | 0.1355        | -               |
| 1.125  | 900  | 0.1248        | -               |
| 1.1875 | 950  | 0.101         | -               |
| 1.25   | 1000 | 0.1135        | -               |
| 1.3125 | 1050 | 0.1046        | -               |
| 1.375  | 1100 | 0.1127        | -               |
| 1.4375 | 1150 | 0.0818        | -               |
| 1.5    | 1200 | 0.1068        | -               |
| 1.5625 | 1250 | 0.0803        | -               |
| 1.625  | 1300 | 0.1013        | -               |
| 1.6875 | 1350 | 0.1018        | -               |
| 1.75   | 1400 | 0.032         | -               |
| 1.8125 | 1450 | 0.0836        | -               |
| 1.875  | 1500 | 0.0539        | -               |
| 1.9375 | 1550 | 0.0573        | -               |
| 2.0    | 1600 | 0.0381        | -               |
| 2.0625 | 1650 | 0.0829        | -               |
| 2.125  | 1700 | 0.0654        | -               |
| 2.1875 | 1750 | 0.0344        | -               |
| 2.25   | 1800 | 0.0721        | -               |
| 2.3125 | 1850 | 0.046         | -               |
| 2.375  | 1900 | 0.0359        | -               |
| 2.4375 | 1950 | 0.0238        | -               |
| 2.5    | 2000 | 0.1032        | -               |
| 2.5625 | 2050 | 0.0442        | -               |
| 2.625  | 2100 | 0.0729        | -               |
| 2.6875 | 2150 | 0.0174        | -               |
| 2.75   | 2200 | 0.0645        | -               |
| 2.8125 | 2250 | 0.0575        | -               |
| 2.875  | 2300 | 0.0476        | -               |
| 2.9375 | 2350 | 0.0388        | -               |
| 3.0    | 2400 | 0.0227        | -               |

### Framework Versions
- Python: 3.12.12
- SetFit: 1.0.3
- Sentence Transformers: 3.1.1
- Transformers: 4.40.0
- PyTorch: 2.11.0
- Datasets: 3.2.0
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->