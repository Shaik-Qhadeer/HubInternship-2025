**I. INTRODUCTION**

The lens is a unique organ that lacks arterial or venous blood circulation and exhibits relatively low levels of cellular metabolism. The elongated lens fibre cells lack organelles such as nuclei, mitochondria and ribosomes, and are packed with a high concentration of distinctive proteins — the lens crystallins31. The crystallins are transparent and have a high refractive index. The lens proteins are not replaced or degraded, and the continued transparency of the lens depends on crystallins maintaining their native structures and solubility throughout the human lifetime. Loss of their native fold renders crystallins prone to aggregation into high molecular-weight complexes, which results in cataract.\[1\]

Cataracts develop gradually and are typically classified into different stages based on their appearance, density, and impact on vision. Understanding these stages is essential for proper diagnosis, treatment planning, and tracking disease progression. The common phases of cataract development include:

1.Incipient (Early Stage): Slight clouding begins in the lens, often with no noticeable vision problems. Often referred to as immature cataracts – Grade I, showing soft white or greenish-yellow opacity.

2.Immature Cataract (Moderate Stage): Lens becomes more visibly cloudy, affecting vision and causing glare or blurriness. Includes Grade II, where the lens is soft-medium with yellowish discoloration.

3.Mature Cataract (Advanced Stage): The lens is completely opaque, leading to significant vision loss or blindness. Involves Grade III and Grade IV:

·        Grade III: Medium-hard amber-colored lens

·        Grade IV: Hard brownish or dark brown lens (sometimes called "hyper-mature").

4.Hypermature Cataract (Overripe Stage): The lens starts to shrink or leak proteins, causing inflammation and complications. Rare in early-detected cases, but critical if surgery is delayed.

 

**Types and Contributing Factors of Cataracts\[2\]:**

Cataracts may arise as a result of numerous causes, which vary from hereditary factors to aging and environmental factors. The following are the primary types and related contributing factors:

Congenital Cataract:

This type of cataract is congenital and can occur in one or both eyes. It has been strongly associated with maternal complications during pregnancy like nutritional deficiencies, infection with Rubella and Rubeola, and less oxygen supply due to placental issues. There is a need for early detection in order to avoid permanent impairment of vision in infants.

Age-Related (Senile) Cataract:

The most common type, age-related cataract, is a natural consequence of aging. As individuals age, proteins in the lens slowly deteriorate and aggregate to result in clouding of the lens as well as cumulative vision loss.

Subcapsular Cataract:

These cataracts develop under the front (anterior) or back (posterior) lens capsule. Posterior subcapsular cataracts are especially effecting on vision, since they are close to the eye's visual axis. Their victims complain of glare, halos, and photophobia. During an examination, they are seen as plaques or granular lesions and can be mistaken for vacuolated cells (also referred to as bladder or Wedl cells).

Nuclear Sclerotic Cataract:

Prevalent in older people, this form entails hardening and discoloration of the center of the lens (nucleus). It's usually associated with myopia and can result in a transient improvement in near vision, often called the second sight.As the disease progresses, the lens can acquire a brown or even black color, and minor changes in vision can be noted on clinical examination.

Cortical Cataract:

These cataracts form in the lens cortex and are marked by wedge or radial spoke opacities, usually occurring in the lower inner (inferonasal) quadrant. Due to fluid accumulation among lens fibers, they cause light scatter, glare, and photophobia, particularly in bright light.

Christmas Tree Cataract:

A uncommon and aesthetically pleasing form, the Christmas tree cataract occurs as multi-colored, needle-shaped crystals in deeper portions of the lens. Although rare, it occurs in some systemic diseases and needs meticulous slit-lamp examination to be diagnosed.

 

 

**II. LITERATURE REVIEW**

 

The paper \[3\] employs GhostYOLOv8n that adds a lightweight and effective method for cataract detection, incorporating GhostConv layers to improve feature extraction with less computational cost. It possesses 1.6 million parameters with an accuracy of 98.1%, detection rate of 191.6 FPS, and low memory requirement (3.7 MB). GhostYOLOv8n greatly enhances performance and real-time deployment. The proposed framework begins with a collecting dataset through Roboflow: https://universe.roboflow.com/ cdacm/eye-health3 which had 427 diverse images, initially imbalanced between normal and cataract. Augmentation grew it to 1,708 images (920 cataract, 788 normal). Then the dataset is split into training (80%), testing (10%), and validation (10%) subsets. The proposed model is then trained and validated using this dataset. Then, the model is deployed on a Jetson Nano board, equipped with a Lenovo CMOS camera. The Jetson Nano processes real-time input, and the output, displayed on an attached desktop, indicates whether the image represents a normal or cataract condition, along with a confidence score.                                                                                                         Future work of this work emphasis on the validation of GhostYoloV8n in multiple clinical contexts and its growing integration into telemedicine and mobile health systems to enhance early cataract access diagnosis, especially in remote and disadvantaged communities.

 The research \[4\] created a deep learning system (DLS) for visually impaired cataracts screening with fundus images. The research overcame the challenge of effective screening in populations that are disadvantaged. They utilized 8,395 high-quality fundus  images among 5,245 people, classified as non-cataracts (3,569), mild cataracts (3,245), and visually impaired cataracts (1,581). The main dataset consisted of 6,997 images among 4,346 subjects sourced from Zhejiang Eye Hospital, Wenzhou (ZEHWZ). Two  external data sets were employed for testing: 1,097 images from ZEHHZ and 301 images from NEH, all of which were chosen employing the same inclusion and exclusion criteria.During preprocessing, images were converted to 224×224 pixels and normalized.                                                                                                                                         Data augmentation was applied to increase dataset diversity.Three CNN architectures—DenseNet121, ResNet50, and Inception V3—were trained and evaluated. DenseNet121 performed best, achieving AUC scores of 0.998 (non-cataracts), 0.938 (mild cataracts), and 0.937 (visually impaired cataracts). The DLS outperformed cataract specialists in detecting visually impaired cataracts and showed comparable accuracy in other classifications.

The researchers \[5\] gathered data from Bangalore Medical College and Minto Eye Hospital aimed at creating an automated deep learning model for cataract stage identification and classification—normal, immature, and mature—based on eye images. The system applied transfer learning using Inception V3, a pre-trained CNN architecture, to enhance feature extraction and classification accuracy.The database comprised preprocessed images that were resized to 500×500 pixels, and cataracts were graded into four grades, categorized under immature (Grades I & II) and mature (Grades III & IV). The model was trained on and tested against different databases to prevent overfitting.The model proposed a 87.5% accuracy and was able to differentiate between the three stages of cataracts effectively.

 

 

**REFERENCES**

 

\[1\]Lam, Dennis & Rao, Srinivas & Ratra, Vineet & Liu, Yizhi & Mitchell, Paul & King, Jonathan & Tassignon, Marie-Jose & Jonas, Jost & Pang, Chi & Chang, David. (2015). Cataract. Nature Reviews Disease Primers. 1\. 15014\. 10.1038/nrdp.2015.14.

\[2\] Nizami AA, Gurnani B, Gulani AC. Cataract. \[Updated 2024 Feb 27\]. In: StatPearls \[Internet\]. Treasure Island (FL): StatPearls Publishing; 2025 Jan-. Available from: [https://www.ncbi.nlm.nih.gov/books/NBK539699/](https://www.ncbi.nlm.nih.gov/books/NBK539699/)

\[3\]Lahari, P.L. & Poola, Rahul & Gorrepati, Leela & Yellampalli, Siva. (2025). Real-Time Cataract Diagnosis with GhostYOLO: A GhostConv-enhanced YOLO Model. Engineering, Technology and Applied Science Research. 15\. 22945-22952. 10.48084/etasr.10760.

\[4\] Xie H, Li Z, Wu C, Zhao Y, Lin C, Wang Z, Wang C, Gu Q, Wang M, Zheng Q, Jiang J, Chen W. Deep learning for detecting visually impaired cataracts using fundus images. Front Cell Dev Biol. 2023 Jul 28;11:1197239. doi: 10.3389/fcell.2023.1197239. PMID: 37576595; PMCID: PMC10416247.

\[5\] Manjunath, Sahana & S Nath, Gowrishankar. (2019). Identification and Classification of Cataract Stages in Old Age People Using Deep Learning Algorithm. International Journal of Innovative Technology and Exploring Engineering. 8\. 2767-2772. 10.35940/ijitee.J9582.0881019.

\[6\] Epidemiology of Risk Factors for Age-Related Cataract ([https://www.sciencedirect.com/science/article/pii/S0039625705801109/pdf?md5=424407ecd173048d1641fa0b00396d68\&pid=1-s2.0-S0039625705801109-main.pdf](https://www.sciencedirect.com/science/article/pii/S0039625705801109/pdf?md5=424407ecd173048d1641fa0b00396d68&pid=1-s2.0-S0039625705801109-main.pdf))

 

 

