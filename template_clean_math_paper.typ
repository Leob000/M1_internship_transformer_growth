#import "@preview/great-theorems:0.1.2": great-theorems-init, mathblock, proofblock
#import "@preview/rich-counters:0.2.2": rich-counter
#import "@preview/i-figured:0.2.4": reset-counters, show-equation

#let currently_writing = true

#let below_math = 8pt
#let above_math = 9pt
#let my_text_size = 11pt
#let my_margins = (left: 25mm, right: 25mm, top: 25mm, bottom: 30mm)
#let heading1_size = 14pt
#let heading2_size = 12pt
#let heading3_4_size = 11pt

// Settings to see better when writing the paper
#if currently_writing {
  my_text_size = 14pt
  my_margins = (left: 5mm, right: 5mm, top: 5mm, bottom: 10mm)
  heading1_size = 17pt
  heading2_size = 15pt
  heading3_4_size = 14pt
}


// counter for mathblocks
#let theoremcounter = rich-counter(
  identifier: "mathblocks",
  inherited_levels: 1,
)

#let my-mathblock = mathblock.with(
  counter: theoremcounter,
  breakable: false,
  titlix: title => [(#title):],
  // suffix: [#h(1fr) $triangle.r$],
)

// theorem etc. settings
#let theorem = my-mathblock(
  blocktitle: "Theorem",
  bodyfmt: text.with(style: "italic"),
)

#let proposition = my-mathblock(
  blocktitle: "Proposition",
  bodyfmt: text.with(style: "italic"),
)

#let corollary = my-mathblock(
  blocktitle: "Corollary",
  bodyfmt: text.with(style: "italic"),
)

#let lemma = my-mathblock(
  blocktitle: "Lemma",
  bodyfmt: text.with(style: "italic"),
)

#let definition = my-mathblock(
  blocktitle: "Definition",
  bodyfmt: text.with(style: "italic"),
)

#let remark = my-mathblock(
  blocktitle: [_Remark_],
  // bodyfmt: text.with(style: "italic"),
  inset: 10pt,
  stroke: rgb("#efe6ff"),
  radius: 5pt,
)

#let example = my-mathblock(blocktitle: [_Example_])

#let question = my-mathblock(blocktitle: [_Question_])

#let proof = proofblock(
  inset: 10pt,
  stroke: rgb("#efe6ff"),
  radius: 5pt,
)

// To also handle content (e.g. something like $dagger$) as affiliation-id,
// cf. https://github.com/typst/typst/issues/2196#issuecomment-1728135476
#let to-string(content) = {
  if type(content) in (int, float, decimal, version, bytes, label, type, str) {
    str(content)
  } else {
    if content.has("text") {
      content.text
    } else if content.has("children") {
      content.children.map(to-string).join("")
    } else if content.has("body") {
      to-string(content.body)
    } else if content == [ ] {
      " "
    }
  }
}

#let template(
  title: "",
  authors: (),
  affiliations: (),
  date: datetime.today().display(),
  abstract: none,
  keywords: (),
  AMS: (),
  heading-color: rgb("#000000"),
  link-color: rgb("#000000"),
  body,
) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(
    margin: my_margins,
    numbering: "1",
    number-align: center,
  )
  set text(font: "New Computer Modern", lang: "en", size: my_text_size)
  show link: set text(fill: link-color)
  show ref: set text(fill: link-color)
  set enum(numbering: "(i)")
  set outline(indent: 2em) // indent: auto does not work well with appendices
  show: great-theorems-init

  // table label on top and not below the table
  show figure.where(kind: table): set figure.caption(position: top)

  // Heading settings.
  set heading(numbering: "1.1")
  show heading.where(level: 1): set text(size: heading1_size, fill: heading-color)
  show heading.where(level: 2): set text(size: heading2_size, fill: heading-color)
  show heading.where(level: 3): set text(size: heading3_4_size, fill: heading-color)
  show heading.where(level: 4): set text(size: heading3_4_size, fill: heading-color)


  // Equation settings.
  // Using i-figured:
  show heading: reset-counters
  show math.equation: show-equation.with(prefix: "eq:", only-labeled: true)

  // Using headcount:
  // show heading: reset-counter(counter(math.equation))
  // set math.equation(numbering: dependent-numbering("(1.1)"))
  set math.equation(supplement: none)
  // show math.equation: box // no line breaks in inline math

  // To have custom size for math:
  // show math.equation: set text(weight: 500, size: 11pt)
  show math.equation: set block(below: below_math, above: above_math)

  line(length: 100%, stroke: 2pt)
  // Title row.
  pad(
    bottom: 4pt,
    top: 4pt,
    align(center)[
      #block(text(weight: 500, fill: heading-color, 1.75em, title))
      #v(1em, weak: true)
    ],
  )
  line(length: 100%, stroke: 2pt)

  // Author information.
  pad(
    top: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1.5em,
      ..authors.map(author => align(center)[
        #let affiliation-id = if "affiliation-id" in author {
          author.affiliation-id
        } else {
          ""
        }
        #if author.keys().contains("orcid") {
          link("http://orcid.org/" + author.orcid)[
            #pad(
              bottom: -8pt,
              grid(
                columns: (8pt, auto, 8pt),
                rows: 10pt,
                [],
                text(black)[*#author.name*#super(to-string(affiliation-id))],
                [
                  #pad(left: 4pt, top: -4pt, image("orcid.svg", width: 8pt))
                ],
              ),
            )
          ]
        } else {
          grid(
            columns: auto,
            rows: 2pt,
            [*#author.name*#super(to-string(affiliation-id))],
          )
        }
      ]),
    ),
  )

  // Affiliation information.
  pad(
    top: 0.5em,
    x: 2em,
    if affiliations != none {
      for affiliation in affiliations {
        align(center)[
          #super(to-string(affiliation.id))#affiliation.name
        ]
      }
    },
  )

  align(center)[#date]

  // Abstract.
  if abstract != none {
    pad(
      x: 3em,
      top: 1em,
      bottom: 0.4em,
      align(center)[
        #heading(
          outlined: false,
          numbering: none,
          text(0.85em, smallcaps[Abstract]),
        )
        #set par(justify: true)
        #set text(hyphenate: false)

        #abstract
      ],
    )
  }

  // Keywords
  if keywords.len() > 0 {
    [*_Keywords_*. #h(0.3cm)] + keywords.map(str).join(" · ")
    linebreak()
  }
  // AMS
  if AMS.len() > 0 {
    [*AMS subject classification*. #h(0.3cm)] + AMS.map(str).join(", ")
  }
  // Main body.
  set par(
    spacing: 0.6em,
    first-line-indent: 1.2em,
    justify: true,
    // leading: 0.6em,
  )
  set text(hyphenate: false)

  body
}

#let appendices(body) = {
  counter(heading).update(0)
  counter("appendices").update(1)

  set heading(
    numbering: (..nums) => {
      let vals = nums.pos()
      let value = "ABCDEFGHIJ".at(vals.at(0) - 1)
      if vals.len() == 1 {
        return "Appendix " + value
      } else {
        return value + "." + nums.pos().slice(1).map(str).join(".")
      }
    },
  )
  [#pagebreak() #body]
}
