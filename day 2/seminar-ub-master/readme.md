
# Seminar Data Mining Matematika UB
Repo ini dibuat untuk memenuhi kebutuhan peserta untuk belajar dan mengerti beberapa algoritma machine learning yang digunakan dalam metode data mining. 

#### -- Project Status: [~in progress]
![](img/process.png)
#### TO DO 
- [ ] make pipeline for latihan_sendiri 
- [ ] answer sheet for latihan_Sendiri
- [x] slides 
  - [x] algorithm pseudocode 
  - [x] visualization 
- [x] kasih capture dari source kmeans, svm, dan hierarchical clustering 
- [x] source code dari sklearn buat ke slidenya pak muklash 
  - [x] svm
  - [x] hierarchical 
  - [x] k-means
  - [x] gmm 
- [x] buat animasi iterasi dari kmeans  


### Contributor
* [Dr. Imam Mukhlash, S.Si, MT.](https://www.researchgate.net/profile/Imam_Mukhlash)
* [Sumihar Christian](github.com/svmihar)

### Methods Used
* Klasifikasi
  * SVM 
    * linear
    * non-linear
  * Bayesian Network 
    * Multinomial Navie Bayes
* Clustering
  * K-Means
    * Cek Konvergensi
  * Gaussian Mixture
* Data Visualization Methods using Matplotlib

### Technologies
* Python 
  * pandas
  * matplotlib
  * scikit-learn

## Getting Started

### before cloning ensure Anaconda is installed correctly 
1. Download [Anaconda here](Anaconda3-2018.12-Windows-x86_64.exe)
2. ensure your path is right
3. try typing `python` in your command prompt
4. install depedencies: 
   1. `pip install matplotlib sklearn seaborn`

### Anaconda too big? 
1. run cmd with adminstrator rights
2. run this command 
   1. `@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"`
3. after finished run this
   1. `choco install python`
   2. `choco install pip`
4. then install dependencies
   1. `pip install matplotlib sklearn seaborn`

5. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
6. Raw Data is being kept [here](https://github.com/svmihar/seminar-ub/blob/master/xclara.csv) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
7. Data processing/transformation scripts are being kept [here](https://github.com/svmihar/seminar-ub/blob/master/Seminar%20Data%20Mining%20Matematika%20UB.ipynb)
8. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructsions](Link to file)

## Referensi
* [Hands on Machine learning and Tensorflow](http://download.library1.org/main/1637000/1f5f9ed30df4b2547fb85c8c2349840b/Aur%C3%A9lien%20G%C3%A9ron%20-%20Hands-On%20Machine%20Learning%20with%20Scikit-Learn%20and%20TensorFlow_%20Concepts%2C%20Tools%2C%20and%20Techniques%20to%20Build%20Intelligent%20Systems-O%E2%80%99Reilly%20Media%20%282017%29.epub)
* [Towards Data Science](https://towardsdatascience.com/)
