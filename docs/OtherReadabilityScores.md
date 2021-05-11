# Other Readability Scores

今回予測対象となっている readability score (target) が存在するが、その他にも既に readability score が世の中に存在するらしいのでそれらについて特徴や計算方法についてまとめる。

## flesch_re (Flesch reading ease)

- 値が大きいほど読みやすい

$$
206.835 - 1.015 * \frac{total\ words}{total\ sentences} - 84.6 * \frac{total\ syllables}{total\ words}
$$

## flesch_kg (Flesch–Kincaid grade level)

- [アメリカのグレードレベル](https://en.wikipedia.org/wiki/Education_in_the_United_States#School_grades)に対応する数値
- 値が小さいほど読みやすい

$$
0.39 * \frac{total\ words}{total\ sentences} + 11.8 * \frac{total\ syllables}{total\ words} - 15.59
$$

## fog_scale (Gunning fog index)

- 値が小さいほど読みやすい

$$
0.4 * \bigl((\frac{words}{sentences}) + 100 * (\frac{complex\ words}{words}) \bigl)
$$

## automated_r (Automated readability index)

- 値が小さいほど読みやすい
- 音節を活用していないのが特徴的

$$
4.71 * (\frac{characters}{words}) + 0.5 * (\frac{words}{sentences}) - 21.43
$$

## coleman (Coleman–Liau index)

- 値が小さいほど読みやすい
- 音節を利用していないのが特徴的

$$
0.0588 * (\frac{total\ characters}{total\ words} * 100) - 0.296 * (\frac{total\ sentences}{total\ words} * 100) - 15.8
$$

## linsear (Linsear Write)

$$
\frac{r}{2},\quad if\ r > 20 \\
\frac{r}{2} - 1,\quad if\ r \leq 20
$$

## text_standard

- 様々な Readability Score からグレードを出力する
- https://textstat.readthedocs.io/en/latest/
  - '12th and 13th grade', e.g

## Glossary

- syllables: 音節。英語特有のもの
