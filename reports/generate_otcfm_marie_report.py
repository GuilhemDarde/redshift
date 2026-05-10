import os
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = "/Users/a33672/Desktop/Stage AstroAI/redshift"
DOWNLOADS = "/Users/a33672/Downloads"
OUT_DIR = os.path.join(ROOT, "reports")
PDF_PATH = os.path.join(OUT_DIR, "rapport_reunion_otcfm_i2i_marie.pdf")
MD_PATH = os.path.join(OUT_DIR, "rapport_reunion_otcfm_i2i_marie.md")


def read_multi_csv(path, subset=False):
    index_col = [0, 1] if subset else 0
    df = pd.read_csv(path, header=[0, 1], index_col=index_col)
    if ("Unnamed: 1_level_0", "Unnamed: 1_level_1") in df.columns:
        df = df.drop(columns=[("Unnamed: 1_level_0", "Unnamed: 1_level_1")])
    return df


def metric(df, ablation, metric_name, stat="mean", subset=None):
    if subset is None:
        return float(df.loc[ablation, (metric_name, stat)])
    return float(df.loc[(ablation, subset), (metric_name, stat)])


def pct(new, ref):
    return 100.0 * (new - ref) / ref


def fmt(value, ndigits=4):
    return f"{value:.{ndigits}f}"


def fmt_pct(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def p(text, style):
    return Paragraph(text, style)


def make_table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d0d0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#444444"),
        ),
    }


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    soft_global = read_multi_csv(os.path.join(DOWNLOADS, "marie_augmented_i2i_soft_10k_seed_summary.csv"))
    combo_global = read_multi_csv(os.path.join(DOWNLOADS, "marie_augmented_i2i_soft_10k_combo_seed_summary.csv"))
    combo_subset = read_multi_csv(os.path.join(DOWNLOADS, "marie_augmented_i2i_soft_10k_combo_seed_subset_summary.csv"), subset=True)
    final_subset = read_multi_csv(os.path.join(DOWNLOADS, "marie_augmented_i2i_soft_10k_visualfiltered_seed_subset_summary.csv"), subset=True)
    visual_band = pd.read_csv(os.path.join(DOWNLOADS, "visual_band_summary.csv"))
    visual_filter = pd.read_csv(os.path.join(DOWNLOADS, "visual_filter_summary.csv")).iloc[0]

    S = styles()
    story = []
    md = []

    title = "Rapport d'avancement - augmentation OT-CFM image-to-image pour la baseline de Marie"
    story += [p(title, S["title"])]
    story += [p(f"Date: {date.today().strftime('%d/%m/%Y')}", S["small"])]
    story += [Spacer(1, 0.2 * cm)]
    md += [f"# {title}", "", f"Date: {date.today().strftime('%d/%m/%Y')}", ""]

    intro = (
        "L'objectif de cette série d'expériences était de tester si des augmentations générées par notre OT-CFM "
        "pouvaient améliorer la prédiction du redshift sur la baseline de Marie, en particulier pour les galaxies "
        "en zone de faible densité. J'ai volontairement gardé la validation et le test uniquement sur de vraies "
        "images afin de ne pas mesurer un gain artificiel sur des données synthétiques."
    )
    story += [p("1. Objectif", S["h1"]), p(intro, S["body"])]
    md += ["## 1. Objectif", intro, ""]

    method = [
        "Le modèle évalué est uniquement la baseline de Marie. Notre G-CNN/MDN n'est pas utilisé comme modèle final.",
        "La faible densité est définie par une densité kNN sur RA/DEC, avec k=10 et un seuil au quantile bas 20% du train.",
        "L'augmentation i2i suit l'idée DA-Fusion/SDEdit, mais dans le cadre OT-CFM: image réelle -> inversion partielle dans le flot -> petite perturbation -> reconstruction conditionnée.",
        "Les conditions physiques restent celles de la galaxie source: redshift spectroscopique, magnitude i, couleurs et morphologie.",
        "Les métriques principales sont bias, sigma_NMAD, RMSE et taux d'outliers, avec une lecture globale et par sous-échantillon de densité.",
    ]
    story += [p("2. Protocole expérimental", S["h1"])]
    for item in method:
        story += [p("- " + item, S["bullet"])]
    md += ["## 2. Protocole expérimental", *[f"- {x}" for x in method], ""]

    story += [p("3. Itérations et difficultés rencontrées", S["h1"])]
    md += ["## 3. Itérations et difficultés rencontrées", ""]

    iter1 = (
        "J'ai d'abord lancé une génération plus agressive avec inversion/reconstruction CFM et interpolation latente "
        "(t0=0.55, noise_scale=0.08). Cette version a produit beaucoup d'images acceptées par le premier filtre "
        "photométrique, mais les performances de Marie se sont dégradées. Après correction de l'évaluation de densité "
        "kNN, les ablations i2i/interp étaient moins bonnes que les augmentations classiques en sigma_NMAD. "
        "Cette étape a montré que le simple fait de générer beaucoup d'images plausibles n'était pas suffisant."
    )
    story += [p("Iteration 1 - CFM i2i/interpolation agressif", S["h2"]), p(iter1, S["body"])]
    md += ["### Iteration 1 - CFM i2i/interpolation agressif", iter1, ""]

    soft_rows = [
        ["Ablation", "sigma_NMAD", "RMSE", "Outliers", "Bias"],
        ["real", fmt(metric(soft_global, "real", "sigma_nmad")), fmt(metric(soft_global, "real", "rmse")), fmt(metric(soft_global, "real", "outlier_rate")), fmt(metric(soft_global, "real", "bias"))],
        ["classic", fmt(metric(soft_global, "classic", "sigma_nmad")), fmt(metric(soft_global, "classic", "rmse")), fmt(metric(soft_global, "classic", "outlier_rate")), fmt(metric(soft_global, "classic", "bias"))],
        ["i2i soft", fmt(metric(soft_global, "i2i", "sigma_nmad")), fmt(metric(soft_global, "i2i", "rmse")), fmt(metric(soft_global, "i2i", "outlier_rate")), fmt(metric(soft_global, "i2i", "bias"))],
    ]
    iter2 = (
        "J'ai ensuite réduit la force de l'augmentation avec t0=0.25 et noise_scale=0.02. Sur 10 000 candidats, "
        "8 057 ont passé le filtre photométrique. Sur trois seeds, i2i soft est devenu compétitif en sigma_NMAD, "
        "mais il gardait plus d'outliers et plus de biais que les augmentations classiques."
    )
    story += [p("Iteration 2 - i2i soft", S["h2"]), p(iter2, S["body"]), make_table(soft_rows)]
    story += [Spacer(1, 0.15 * cm)]
    md += ["### Iteration 2 - i2i soft", iter2, "", md_table(soft_rows[0], soft_rows[1:]), ""]

    combo_rows = [
        ["Ablation", "global sigma", "low-density sigma", "global outliers", "low-density outliers"],
        ["classic", fmt(metric(combo_subset, "classic", "sigma_nmad", subset="global")), fmt(metric(combo_subset, "classic", "sigma_nmad", subset="low_density")), fmt(metric(combo_subset, "classic", "outlier_rate", subset="global")), fmt(metric(combo_subset, "classic", "outlier_rate", subset="low_density"))],
        ["i2i", fmt(metric(combo_subset, "i2i", "sigma_nmad", subset="global")), fmt(metric(combo_subset, "i2i", "sigma_nmad", subset="low_density")), fmt(metric(combo_subset, "i2i", "outlier_rate", subset="global")), fmt(metric(combo_subset, "i2i", "outlier_rate", subset="low_density"))],
        ["classic_i2i", fmt(metric(combo_subset, "classic_i2i", "sigma_nmad", subset="global")), fmt(metric(combo_subset, "classic_i2i", "sigma_nmad", subset="low_density")), fmt(metric(combo_subset, "classic_i2i", "outlier_rate", subset="global")), fmt(metric(combo_subset, "classic_i2i", "outlier_rate", subset="low_density"))],
    ]
    iter3 = (
        "La combinaison naive real + classic + i2i soft n'a pas donné le meilleur des deux mondes. "
        "Elle a réduit certains outliers par rapport a i2i seul, mais elle a perdu le gain en sigma_NMAD. "
        "Cette difficulté m'a poussé a regarder les images directement au lieu de rester uniquement sur les métriques."
    )
    story += [p("Iteration 3 - combinaison naive classic_i2i", S["h2"]), p(iter3, S["body"]), make_table(combo_rows)]
    story += [Spacer(1, 0.15 * cm)]
    md += ["### Iteration 3 - combinaison naive classic_i2i", iter3, "", md_table(combo_rows[0], combo_rows[1:]), ""]

    visual = (
        "L'inspection visuelle bande par bande a montré que les images i2i étaient globalement plausibles, mais très "
        "conservatrices et légèrement assombries. Plusieurs exemples montraient du lissage, des modifications du fond "
        "ou des variations trop fortes dans certaines bandes. J'ai donc ajouté un second filtre basé sur les ratios "
        "de flux source/augmentation, la L1 relative et la corrélation pixel par bande."
    )
    story += [p("Iteration 4 - inspection visuelle et filtrage bande par bande", S["h2"]), p(visual, S["body"])]
    md += ["### Iteration 4 - inspection visuelle et filtrage bande par bande", visual, ""]

    filter_text = (
        f"Le filtre visuel a conservé {int(visual_filter['accepted_by_visual_filter'])} images sur "
        f"{int(visual_filter['evaluated_candidates'])}, soit {visual_filter['acceptance_rate_pct']:.1f}% des "
        "augmentations i2i déjà acceptées photométriquement."
    )
    story += [p(filter_text, S["body"])]
    md += [filter_text, ""]

    band_rows = [["Bande", "Flux ratio median", "p16-p84", "L1 mediane", "Correlation mediane"]]
    for _, row in visual_band.iterrows():
        band_rows.append([
            row["band"],
            fmt(row["median_flux_ratio"], 3),
            f"{row['p16_flux_ratio']:.3f}-{row['p84_flux_ratio']:.3f}",
            fmt(row["median_relative_l1"], 3),
            fmt(row["median_correlation"], 3),
        ])
    story += [make_table(band_rows)]
    story += [Spacer(1, 0.15 * cm)]
    md += [md_table(band_rows[0], band_rows[1:]), ""]

    final_global_rows = [["Ablation", "sigma_NMAD", "RMSE", "Outliers", "Bias"]]
    for ab in ["real", "classic", "i2i", "classic_i2i"]:
        final_global_rows.append([
            ab,
            fmt(metric(final_subset, ab, "sigma_nmad", subset="global")),
            fmt(metric(final_subset, ab, "rmse", subset="global")),
            fmt(metric(final_subset, ab, "outlier_rate", subset="global")),
            fmt(metric(final_subset, ab, "bias", subset="global")),
        ])
    final_low_rows = [["Ablation", "sigma_NMAD", "RMSE", "Outliers", "Bias"]]
    for ab in ["real", "classic", "i2i", "classic_i2i"]:
        final_low_rows.append([
            ab,
            fmt(metric(final_subset, ab, "sigma_nmad", subset="low_density")),
            fmt(metric(final_subset, ab, "rmse", subset="low_density")),
            fmt(metric(final_subset, ab, "outlier_rate", subset="low_density")),
            fmt(metric(final_subset, ab, "bias", subset="low_density")),
        ])

    final_text = (
        "Après ce filtrage, la version classic_i2i devient la meilleure en sigma_NMAD, globalement et en faible densité. "
        f"En faible densité, sigma_NMAD passe de {metric(final_subset, 'classic', 'sigma_nmad', subset='low_density'):.5f} "
        f"pour classic a {metric(final_subset, 'classic_i2i', 'sigma_nmad', subset='low_density'):.5f} pour classic_i2i "
        f"({fmt_pct(pct(metric(final_subset, 'classic_i2i', 'sigma_nmad', subset='low_density'), metric(final_subset, 'classic', 'sigma_nmad', subset='low_density')))}). "
        "En revanche, la RMSE et les outliers ne s'améliorent pas aussi nettement. La méthode réduit donc surtout "
        "l'erreur centrale robuste, mais elle ne règle pas encore les cas catastrophiques."
    )
    story += [p("4. Résultats finaux après filtrage visuel", S["h1"]), p(final_text, S["body"])]
    story += [p("Résultats globaux", S["h2"]), make_table(final_global_rows)]
    story += [Spacer(1, 0.15 * cm), p("Résultats faible densité", S["h2"]), make_table(final_low_rows)]
    md += ["## 4. Résultats finaux après filtrage visuel", final_text, "", "### Résultats globaux", md_table(final_global_rows[0], final_global_rows[1:]), "", "### Résultats faible densité", md_table(final_low_rows[0], final_low_rows[1:]), ""]

    conclusion = (
        "Ma conclusion actuelle est que l'OT-CFM i2i n'est pas assez fiable seul, mais qu'il devient utile comme "
        "complément aux augmentations classiques lorsque les images sont filtrées bande par bande. Le résultat le plus "
        "solide est le gain systématique en sigma_NMAD de classic_i2i visual-filtered par rapport a classic sur trois seeds. "
        "La limite principale est que le gain ne se traduit pas encore par une amélioration claire de la RMSE ou des outliers."
    )
    story += [p("5. Conclusion personnelle", S["h1"]), p(conclusion, S["body"])]
    md += ["## 5. Conclusion personnelle", conclusion, ""]

    next_steps = [
        "Tester le dosage du synthétique filtré: max_synthetic=1000, 2000 et 2908.",
        "Ajouter un filtre de cohérence redshift avec un teacher Marie: rejeter si z_pred(source augmentée) s'éloigne trop du z_spec source.",
        "Ajouter un rejet des champs encombrés, car les voisins et le fond sont parfois modifiés avec la galaxie centrale.",
        "Tester une contrainte de conservation de flux directement dans le CFM, ou un loss photométrique multi-bande.",
        "Faire une analyse par bins de redshift et magnitude pour voir où le gain sigma_NMAD vient réellement.",
    ]
    story += [p("6. Prochaines expériences", S["h1"])]
    for item in next_steps:
        story += [p("- " + item, S["bullet"])]
    md += ["## 6. Prochaines expériences", *[f"- {x}" for x in next_steps], ""]

    story += [PageBreak(), p("Résumé pour la réunion", S["title"])]
    md += ["# Résumé pour la réunion", ""]

    meeting_points = [
        "Je pars de la baseline de Marie, pas de notre modèle G-CNN/MDN.",
        "La question testée est: est-ce qu'une augmentation diffusion/flow ciblée faible densité aide le redshift ?",
        "i2i brut marche mal ou de manière instable: il peut améliorer sigma_NMAD mais augmente les outliers.",
        "L'inspection visuelle a montré un biais de flux et un lissage des images, donc j'ai ajouté un filtre par bandes.",
        "Après filtrage, classic_i2i est le meilleur en sigma_NMAD global et faible densité, mais RMSE/outliers restent moins convaincants.",
        "Donc l'approche est prometteuse comme régularisation/complément, pas encore comme méthode qui bat tout.",
    ]
    story += [p("Message principal", S["h1"])]
    for item in meeting_points:
        story += [p("- " + item, S["bullet"])]
    md += ["## Message principal", *[f"- {x}" for x in meeting_points], ""]

    qa = [
        ("Pourquoi sigma_NMAD est importante ?",
         "Parce qu'elle mesure la dispersion robuste de l'erreur relative en redshift. Elle est moins dominée par quelques échecs catastrophiques que la RMSE."),
        ("Pourquoi la RMSE peut se dégrader alors que sigma_NMAD s'améliore ?",
         "Cela veut dire que les erreurs centrales diminuent, mais que quelques cas difficiles restent mauvais ou deviennent plus mauvais. C'est exactement ce que montrent les outliers."),
        ("Pourquoi i2i seul est mauvais après filtrage ?",
         "Parce qu'il remplace une partie du signal d'entraînement par des images très proches mais pas parfaitement label-preserving. Il est plus utile en complément de classic qu'en stratégie isolée."),
        ("Pourquoi regarder les images ?",
         "Les métriques redshift ne disent pas si une bande a été assombrie, si le fond a changé ou si un voisin a été modifié. Or ces effets peuvent casser la photométrie sans être visibles immédiatement dans une métrique globale."),
        ("Quelle est la différence avec DA-Fusion ?",
         "DA-Fusion fait image-to-image via bruitage partiel et débruitage conditionné. Ici je fais l'analogue dans un OT-CFM: inversion partielle par le flot, petite perturbation, puis reconstruction conditionnée."),
        ("Est-ce une fusion de deux images ?",
         "Non. L'image source est partiellement déplacée dans l'espace latent/flow puis reconstruite. Pour l'interpolation latente, le mélange se fait dans le latent, pas en pixels."),
        ("Comment la faible densité est définie ?",
         "Par kNN sur RA/DEC avec k=10. Je calcule le seuil sur le train au quantile bas 20%, puis je reporte l'évaluation sur test low_density et normal_density."),
        ("Pourquoi ne pas mettre les images synthétiques en validation/test ?",
         "Pour éviter une évaluation artificielle. Le modèle est entraîné avec synthétique, mais validé/testé sur de vraies galaxies uniquement."),
        ("Est-ce publiable ?",
         "A ce stade, plutôt comme résultat exploratoire solide ou workshop astro-ML si on ajoute dosage, teacher filter et validation plus poussée. Pour une conférence principale, les gains actuels sont encore trop modestes et pas uniformes sur RMSE/outliers."),
        ("Quelle est la prochaine expérience la plus importante ?",
         "Tester 1000 et 2000 images i2i filtrées au lieu des 2908, pour garder le gain sigma_NMAD tout en essayant de réduire RMSE et outliers."),
    ]
    story += [p("Questions possibles et réponses", S["h1"])]
    md += ["## Questions possibles et réponses", ""]
    for question, answer in qa:
        story += [p(question, S["h2"]), p(answer, S["body"])]
        md += [f"### {question}", answer, ""]

    SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title=title,
    ).build(story)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(PDF_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
