# Capstone Project

# Identifying Pneumonia by Chest X-Ray Images

![chest-xray-image](./media/636619135583776321-GettyImages-530196490.jpg)

I will build a convolutional neural network (CNN) to identify whether a patient has pneumonia or not by classifying their medical images. Recall shall be the success metric as there is a high risk associated with false negative classification when it comes to human diseases.

This project is adequately scoped and focuses on one specific type of disease rather than targeting multiple diagnosis. Therefore, there is a high chance to generate substentially accurate results.

Pneumonia is an infection that inflames lungs and can be diagnosed by radiologists who view the patient's chest x-rays. Creating an algorithm that provides accurate diagnosis can be beneficial for both patients and medical proffesionals.


## Data Guidelines

Source: https://data.mendeley.com/datasets/rscbjbr9sj/2
Published: 6 Jan 2018 | Version 2 | DOI: 10.17632/rscbjbr9sj.2
Contributor(s): Daniel Kermany, Kang Zhang, Michael Goldbaum

![mendeley-website](./media/mendeley.jpg)

Kaggle link:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

**Train Dataset**

3883 .jpeg images labeled as PNEUMONIA (bacterial).
1349 .jpeg images labeled as NORMAL.

**Test Dataset**

390 .jpeg images labeled as PNEUMONIA (viral).
234 .jpeg images labeled as NORMAL.
