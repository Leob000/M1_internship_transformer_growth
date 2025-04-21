#import "template.typ": *
#import "theorems.typ": *
#show: thmrules

#show: ams-article.with(
  title: [Internship report, Attention growing networks],
  authors: (
    (
      name: "Léo Burgund",
      // department: [Department of Mathematics],
      // organization: [University of Exampleville],
      // location: [Tennessee, TN 59341],
      email: "leo.burgund@gmail.com",
      // url: "math.ue.edu/~jdoe"
    ),
  ),
  // abstract: lorem(100),
  bibliography: bibliography("refs.bib"),
)


= Nomenclature
== Dimensions
- $b$ Batch
- $d_e$ Embedding dimension
- $d_s$ Sequence length
- $d_k$ Query/Keys dimension
- $d_v$ Value dimension
- $h$ Number of heads

== Matrix
In the case of multi-head attention, for each head $i = 1,...,h$, we have:
- Input $X in RR^(d_s times d_e)$
- $W_Q_i in RR^(d_e times d_k / colMath(h,#red)), Q_i:= X W_Q_i in RR^(d_s times d_k / colMath(h,#red))$
- $W_K_i in RR^(d_e times d_k / colMath(h,#red)), K_i:= X W_K_i in RR^(d_s times d_k / colMath(h,#red))$
- $S_i := (Q_i K_i^top ) / sqrt(d_k/colMath(h,#red)) in RR^(d_s times d_s)$
- $A_i := "softmax"_"row" (S)$
- $W_V_i in RR^(d_e times d_v / colMath(h,#red)), V_i:= X W_V_i in RR^(d_s times d_v / colMath(h,#red))$
- $H_i := A_i V_i in RR^(d_s times d_v /colMath(h,#red))$, $H=[H_1,...,H_h] in RR^(d_s times d_v)$
- $W_O in RR^(d_v times d_e)$
- Output $Y:= H W_O + X in RR^(d_s times d_e)$

#remark[
  The number of parameters to learn
  $
    (underbrace(2(d_e (d_k)/h), W_Q_i\,W_K_i) + underbrace(d_e d_v/h ,W_V_i))h + underbrace(d_v d_e , W_O)
  $
  is the same for any $h in NN_+^*$.
]

#remark[
  We can easily consider the bias by augmenting the matrices:
  $
    X' = [X | bold(1)] in RR^(d_s times (d_e+1))\
    H' = [H | bold(1)] in RR^(d_s times (d_v + 1))
  $
  And adding a row of parameters to $W_Q_i, W_K_i, W_V_i, W_O$. For example:
  $
    W'_Q_i= vec(W_Q_i, (b^Q)^top ) in RR^((d_e + 1 ) times d_k / h).
  $
]

= Problem
We study the case where $h=1$.

We are interested in growing the $d_k$ dimension.
We consider the first order approximation, using the functional gradient,
$
  cal(L)(f + diff f(d theta, d cal(A)) )= cal(L) (f)+ ip(nabla_f cal(L)(f), diff f(diff theta, diff cal(A)))+o(norm(diff f(diff theta, diff cal(A)))).
$
// TODO trouver comment ref equations en typst

To avoid the softmax's non linearity, we will consider the gradient with respect to the matrix $S$, just before the softmax.
// TODO develop why

We then have
$
  cal(L)(S + diff S )= cal(L) (S)+ ip(nabla_S cal(L)(S), diff S)+o(norm(diff S))
$
with
$
  diff S= X(W_Q+diff W_Q)(W_K + diff W_K)^top X^top - X W_Q W_K^top X^top.
$

We have the following optimization problem:
$
  arg min_(diff S) ip(nabla_S cal(L)(S), diff S), "such that" norm(diff S) <= gamma
$

// TODO justify gamma
// TODO vérifier les signes
// TODO espace plus grand en typst après la virgule?


// TODO justifier passage de l'un à l'autre
$
  arg min_(diff W_Q, diff W_K) norm(B - X(W_Q+diff W_Q)(W_K+ diff W_K)^top X^top )^2 \
  "with" B :=nabla_S cal(L)(S) + X W_Q W_K^top X^top
$
// TODO rescale par gamma pour la solution

Which is a low rank regression (limited by $d_k$). $B$ is known.

// TODO parler des deux méthodes possibles
We can approximate $X underbrace((W_Q+diff W_Q), d_e times d_k)underbrace((W_K+ diff W_K)^top, d_k times d_e) X^top$ with a truncated SVD, taking the first $d_k$ singular values.

To grow $S$ for the next training iteration, we can instead approximate by

// TODO is it always?
