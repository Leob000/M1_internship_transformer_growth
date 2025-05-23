#import "template_clean_math_paper.typ": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "Brouillon 3",
  authors: (
    (name: "Léo Burgund"),
    // (name: "Author 1", affiliation-id: 1, orcid: "0000-0000-0000-0000"),
  ),
  affiliations: (
    // (id: 1, name: "Affiliation 1, Address 1"),
    // (id: "*", name: "Corresponding author")
  ),
  date: date,
  heading-color: rgb("#000000"),
  link-color: rgb("#000082"),
  // abstract: [This is my abstract...],
  // keywords: ("First keyword", "Second keyword", "etc."),
  // AMS: ("65M70", "65M12"),
)

#let colMath(x, color) = text(fill: color)[$#x$]
#let ip(x, y) = $lr(angle.l #x, #y angle.r)$

= Nomenclature
== Dimensions
- $b$ Mini-batch size
- $d_e$ Embedding dimension
- $d_s$ Sequence length
- $d_k$ Query/Keys dimension
- $d_v$ Value dimension
- $h$ Number of heads

We make the hypothesis that $d_k < d_e < d_s$.

== Matrix operations in a self-attention block
// TODO: Check if h factor is numerator or denominator
In the case of multi head attention, for each head $i = 1,...,h$, we have:
- Input $X in RR^(d_s times d_e)$
- $W_Q_i in RR^(d_e times d_k / colMath(h, #red)), Q_i:= X W_Q_i in RR^(d_s times d_k / colMath(h, #red))$
- $W_K_i in RR^(d_e times d_k / colMath(h, #red)), K_i:= X W_K_i in RR^(d_s times d_k / colMath(h, #red))$
- $S_i := (Q_i K_i^top ) / sqrt(d_k / colMath(h, #red)) in RR^(d_s times d_s)$
- $A_i := "softmax"_"row" (S)$
- $W_V_i in RR^(d_e times d_v / colMath(h, #red)), V_i:= X W_V_i in RR^(d_s times d_v / colMath(h, #red))$
- $H_i := A_i V_i in RR^(d_s times d_v / colMath(h, #red))$, $H=[H_1,...,H_h] in RR^(d_s times d_v)$
- $W_O in RR^(d_v times d_e)$
- Output $Y:= H W_O + X in RR^(d_s times d_e)$

For now, we study the case $h=1$.

#emoji.fire We omit the $1 / sqrt(d_k)$ scaling for the $S$ matrix, it can cause problem with growing, so for growing we will make it a learnable parameter (and initialize it at $1 / sqrt(d_k_"initial")$ ?).

= Problem
Goal:
$
  min_f cal(L) (f).
$
We will study the variations of the loss made by the variations of $S$, with other parameters fixed. Hence we will study
$
  arg min_S cal(L) (S)
$
Let $G:= gradient_(S) cal(L)(S)$.

We have
$
  "rk"(G) <= d_s, space "rk"(S) <=d_k
$


We have the first order approximation:
$
  cal(L) (S+ gamma dif S) =cal(L)(S) + gamma ip(G, dif S) + o(norm((dif S)) )
$

We introduce $gamma$, similar to a step size, and we consider the problem
$
  arg min_(dif S) ip(G, dif S)_F "s.t." norm(dif S) <= gamma and "rk"(dif S) <= d_k
$


Let $Z= W_Q W_K^top$, with $"rk"(Z)<= d_k$ , we then have
$
  S&= X W_Q W_K^top X^top\
  &= X Z X^top \
$
and

#emoji.fire To verify
$
  dif S &= X ( W_Q_(+1)W_K_(+1))X^top - X W_Q W_K^top X^top \
  &= X (Z + dif Z) X^top -X Z X^top \
  &= X dif Z X^top \
$

Hence
$
  ip(G, dif S)_F &= ip(G, X dif Z X^top)_F \
  &= tr(G X dif Z^top X^top) \
  &= ip(X^top G X, dif Z)_F \
$

The problem becomes
$
  arg min_(dif Z) ip(X^top G X, dif Z) "s.t." norm(X dif Z X^top) <= gamma and "rk"(dif Z) <= d_k
$

#emoji.fire Problem with the norm constraint:

We can either
- Solve the problem $min_( X dif Z X^top) ip(G, X dif Z X^top) "s.t." norm(X dif Z X^top)<= gamma and "rk"(dif Z) <= d_k$, expensive but ok.
- Try to relax the norm constraint, but that could cause some space warping? and then $dif Z = - X^top G X$ could not be the best direction? (Then search gamma with a line search)

-> Test both to see if the second works?

= Problem with relaxed norm constraint
Let $dif Z^(0)$ be the best direction for $dif Z$.

We consider we can get $dif Z^(0) = - X^top G X$ from the problem, up to a rank constraint. We will scale with gamma later.

In practice, we could accumulate the $dif Z^(0)$:
$
  dif Z^(0) = EE_X [- X^top G X]
$

Then do a line search, either
$
  lambda^star_"FR"= cal(L) (Z+ lambda dif Z^(0) ) \
  lambda^star_"LR" = cal(L) ((Z + lambda dif Z^(0) )_"LR" )
$

Then get the new weight matrices
$
  W_Q_(+1) ,W_K_(+1) = "SVD"_"LR" (Z + lambda^star dif Z^(0) )
$
== Testing coming from the first order approximation
$
  cal(L) (S + gamma dif S) &= cal(L) (S) + gamma ip(G, dif S) + o(norm(dif S) ) \
  &= cal(L) (S) + gamma ip(G, X dif Z X^top) + o(norm(dif S) ) \
  &= cal(L) (S) + gamma ip(X^top G X, dif Z) + o(norm(dif S) ) \
$
#emoji.fire No constraint, during the first order approx, allows to put the norm constraint after? Something like
$
  arg min_(dif Z) ip(X^top G X, dif Z)_F "s.t." norm(dif Z)_F <= gamma and "rk"(dif Z) <= "rk"( X^top G X)
$

== Idea
$
  norm(dif S)_F &= norm(X dif Z X)_F \
  &<= norm(X)_2^2 norm(dif Z)_F\
$
We also have
$
  sigma_min^2 (X) norm(dif Z)_F<= norm(dif S)_F
$
So if there exist $sigma_min^2 >=0 <==> X "full rank"$, then $norm(dif S)$ has a positive lower bound.



= Full problem
