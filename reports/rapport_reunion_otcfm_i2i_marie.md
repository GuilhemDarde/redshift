# Rapport d'avancement - augmentation OT-CFM image-to-image pour la baseline de Marie

Date: 07/05/2026

## 1. Objectif
L'objectif de cette série d'expériences était de tester si des augmentations générées par notre OT-CFM pouvaient améliorer la prédiction du redshift sur la baseline de Marie, en particulier pour les galaxies en zone de faible densité. J'ai volontairement gardé la validation et le test uniquement sur de vraies images afin de ne pas mesurer un gain artificiel sur des données synthétiques.

## 2. Protocole expérimental
- Le modèle évalué est uniquement la baseline de Marie. Notre G-CNN/MDN n'est pas utilisé comme modèle final.
- La faible densité est définie par une densité kNN sur RA/DEC, avec k=10 et un seuil au quantile bas 20% du train.
- L'augmentation i2i suit l'idée DA-Fusion/SDEdit, mais dans le cadre OT-CFM: image réelle -> inversion partielle dans le flot -> petite perturbation -> reconstruction conditionnée.
- Les conditions physiques restent celles de la galaxie source: redshift spectroscopique, magnitude i, couleurs et morphologie.
- Les métriques principales sont bias, sigma_NMAD, RMSE et taux d'outliers, avec une lecture globale et par sous-échantillon de densité.

## 3. Itérations et difficultés rencontrées

### Iteration 1 - CFM i2i/interpolation agressif
J'ai d'abord lancé une génération plus agressive avec inversion/reconstruction CFM et interpolation latente (t0=0.55, noise_scale=0.08). Cette version a produit beaucoup d'images acceptées par le premier filtre photométrique, mais les performances de Marie se sont dégradées. Après correction de l'évaluation de densité kNN, les ablations i2i/interp étaient moins bonnes que les augmentations classiques en sigma_NMAD. Cette étape a montré que le simple fait de générer beaucoup d'images plausibles n'était pas suffisant.

### Iteration 2 - i2i soft
J'ai ensuite réduit la force de l'augmentation avec t0=0.25 et noise_scale=0.02. Sur 10 000 candidats, 8 057 ont passé le filtre photométrique. Sur trois seeds, i2i soft est devenu compétitif en sigma_NMAD, mais il gardait plus d'outliers et plus de biais que les augmentations classiques.

| Ablation | sigma_NMAD | RMSE | Outliers | Bias |
| --- | --- | --- | --- | --- |
| real | 0.0377 | 0.2605 | 5.0999 | 0.0129 |
| classic | 0.0351 | 0.2555 | 4.4848 | 0.0139 |
| i2i soft | 0.0349 | 0.2589 | 4.8962 | 0.0194 |

### Iteration 3 - combinaison naive classic_i2i
La combinaison naive real + classic + i2i soft n'a pas donné le meilleur des deux mondes. Elle a réduit certains outliers par rapport a i2i seul, mais elle a perdu le gain en sigma_NMAD. Cette difficulté m'a poussé a regarder les images directement au lieu de rester uniquement sur les métriques.

| Ablation | global sigma | low-density sigma | global outliers | low-density outliers |
| --- | --- | --- | --- | --- |
| classic | 0.0352 | 0.0357 | 4.6061 | 4.8319 |
| i2i | 0.0350 | 0.0352 | 5.4599 | 5.5125 |
| classic_i2i | 0.0369 | 0.0373 | 4.6226 | 4.6958 |

### Iteration 4 - inspection visuelle et filtrage bande par bande
L'inspection visuelle bande par bande a montré que les images i2i étaient globalement plausibles, mais très conservatrices et légèrement assombries. Plusieurs exemples montraient du lissage, des modifications du fond ou des variations trop fortes dans certaines bandes. J'ai donc ajouté un second filtre basé sur les ratios de flux source/augmentation, la L1 relative et la corrélation pixel par bande.

Le filtre visuel a conservé 2908 images sur 8057, soit 36.1% des augmentations i2i déjà acceptées photométriquement.

| Bande | Flux ratio median | p16-p84 | L1 mediane | Correlation mediane |
| --- | --- | --- | --- | --- |
| u | 0.972 | 0.962-0.982 | 0.049 | 1.000 |
| g | 0.958 | 0.928-0.980 | 0.169 | 0.994 |
| r | 0.986 | 0.965-1.002 | 0.124 | 0.998 |
| i | 0.985 | 0.962-0.996 | 0.096 | 0.999 |
| z | 0.959 | 0.924-0.984 | 0.099 | 0.999 |
| y | 0.972 | 0.950-0.989 | 0.080 | 0.999 |

## 4. Résultats finaux après filtrage visuel
Après ce filtrage, la version classic_i2i devient la meilleure en sigma_NMAD, globalement et en faible densité. En faible densité, sigma_NMAD passe de 0.03558 pour classic a 0.03470 pour classic_i2i (-2.5%). En revanche, la RMSE et les outliers ne s'améliorent pas aussi nettement. La méthode réduit donc surtout l'erreur centrale robuste, mais elle ne règle pas encore les cas catastrophiques.

### Résultats globaux
| Ablation | sigma_NMAD | RMSE | Outliers | Bias |
| --- | --- | --- | --- | --- |
| real | 0.0365 | 0.2569 | 4.7851 | 0.0103 |
| classic | 0.0352 | 0.2534 | 4.5527 | 0.0128 |
| i2i | 0.0382 | 0.2654 | 5.8446 | 0.0286 |
| classic_i2i | 0.0345 | 0.2576 | 4.6761 | 0.0146 |

### Résultats faible densité
| Ablation | sigma_NMAD | RMSE | Outliers | Bias |
| --- | --- | --- | --- | --- |
| real | 0.0363 | 0.2484 | 4.8932 | 0.0125 |
| classic | 0.0356 | 0.2468 | 4.6345 | 0.0146 |
| i2i | 0.0384 | 0.2611 | 5.9684 | 0.0300 |
| classic_i2i | 0.0347 | 0.2575 | 4.7775 | 0.0157 |

## 5. Conclusion personnelle
Ma conclusion actuelle est que l'OT-CFM i2i n'est pas assez fiable seul, mais qu'il devient utile comme complément aux augmentations classiques lorsque les images sont filtrées bande par bande. Le résultat le plus solide est le gain systématique en sigma_NMAD de classic_i2i visual-filtered par rapport a classic sur trois seeds. La limite principale est que le gain ne se traduit pas encore par une amélioration claire de la RMSE ou des outliers.

## 6. Prochaines expériences
- Tester le dosage du synthétique filtré: max_synthetic=1000, 2000 et 2908.
- Ajouter un filtre de cohérence redshift avec un teacher Marie: rejeter si z_pred(source augmentée) s'éloigne trop du z_spec source.
- Ajouter un rejet des champs encombrés, car les voisins et le fond sont parfois modifiés avec la galaxie centrale.
- Tester une contrainte de conservation de flux directement dans le CFM, ou un loss photométrique multi-bande.
- Faire une analyse par bins de redshift et magnitude pour voir où le gain sigma_NMAD vient réellement.

# Résumé pour la réunion

## Message principal
- Je pars de la baseline de Marie, pas de notre modèle G-CNN/MDN.
- La question testée est: est-ce qu'une augmentation diffusion/flow ciblée faible densité aide le redshift ?
- i2i brut marche mal ou de manière instable: il peut améliorer sigma_NMAD mais augmente les outliers.
- L'inspection visuelle a montré un biais de flux et un lissage des images, donc j'ai ajouté un filtre par bandes.
- Après filtrage, classic_i2i est le meilleur en sigma_NMAD global et faible densité, mais RMSE/outliers restent moins convaincants.
- Donc l'approche est prometteuse comme régularisation/complément, pas encore comme méthode qui bat tout.

## Questions possibles et réponses

### Pourquoi sigma_NMAD est importante ?
Parce qu'elle mesure la dispersion robuste de l'erreur relative en redshift. Elle est moins dominée par quelques échecs catastrophiques que la RMSE.

### Pourquoi la RMSE peut se dégrader alors que sigma_NMAD s'améliore ?
Cela veut dire que les erreurs centrales diminuent, mais que quelques cas difficiles restent mauvais ou deviennent plus mauvais. C'est exactement ce que montrent les outliers.

### Pourquoi i2i seul est mauvais après filtrage ?
Parce qu'il remplace une partie du signal d'entraînement par des images très proches mais pas parfaitement label-preserving. Il est plus utile en complément de classic qu'en stratégie isolée.

### Pourquoi regarder les images ?
Les métriques redshift ne disent pas si une bande a été assombrie, si le fond a changé ou si un voisin a été modifié. Or ces effets peuvent casser la photométrie sans être visibles immédiatement dans une métrique globale.

### Quelle est la différence avec DA-Fusion ?
DA-Fusion fait image-to-image via bruitage partiel et débruitage conditionné. Ici je fais l'analogue dans un OT-CFM: inversion partielle par le flot, petite perturbation, puis reconstruction conditionnée.

### Est-ce une fusion de deux images ?
Non. L'image source est partiellement déplacée dans l'espace latent/flow puis reconstruite. Pour l'interpolation latente, le mélange se fait dans le latent, pas en pixels.

### Comment la faible densité est définie ?
Par kNN sur RA/DEC avec k=10. Je calcule le seuil sur le train au quantile bas 20%, puis je reporte l'évaluation sur test low_density et normal_density.

### Pourquoi ne pas mettre les images synthétiques en validation/test ?
Pour éviter une évaluation artificielle. Le modèle est entraîné avec synthétique, mais validé/testé sur de vraies galaxies uniquement.

### Est-ce publiable ?
A ce stade, plutôt comme résultat exploratoire solide ou workshop astro-ML si on ajoute dosage, teacher filter et validation plus poussée. Pour une conférence principale, les gains actuels sont encore trop modestes et pas uniformes sur RMSE/outliers.

### Quelle est la prochaine expérience la plus importante ?
Tester 1000 et 2000 images i2i filtrées au lieu des 2908, pour garder le gain sigma_NMAD tout en essayant de réduire RMSE et outliers.
