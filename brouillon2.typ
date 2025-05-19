#import "template_clean_math_paper.typ": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "Brouillon 2",
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

= Problem
Goal:
$
  min_f cal(L) (f).
$
We will study the variations of the loss made by the variations of $S$, with other parameters fixed. Hence we will study
$
  arg min_S cal(L) (S)
$
with
$
  S= X W_Q W_K^top X^top
$

First order approximation:
$
  cal(L) (S+dif S) =cal(L)(S) + ip(G, dif S) + o(norm((dif S)) )
$
with $G = gradient_(S) cal(L)(S)$, and
$
  dif S &= X(W_Q + dif W_Q)(W_K + dif W_K)^top X^top -X W_Q W_K^top X^top \
  &= X W_Q dif W_K^top X^top +X dif W_Q W_K^top X^top +X dif W_Q dif W_K^top X^top \
  &= X(W_Q dif W_K^top +dif W_Q W_K^top ) X^top +o(norm(dif W_Q) dot norm(dif W_K) )\
$<dS1>
We define
$
  dif S_"linear"&:= X(W_Q dif W_K^top +dif W_Q W_K^top ) X^top\
  dif S_"full" &:= X W_Q dif W_K^top X^top +X dif W_Q W_K^top X^top +X dif W_Q dif W_K^top X^top \
$




---

We will attempt to resolve the following problem:
$
  arg min_(dif S) ip(G, dif S) space "s.t." norm(dif S) <=gamma
$
with $gamma in RR_+$.

$gamma$ is similar to the learning rate, and constrains $dif S$ to respect the first order approximation.

---

The solution $dif S$ has a norm $norm(dif S) =gamma$ when there exists a $dif S$ such that $ip(G, dif S) <= 0$.

We make the hypothesis that we can always find such a $dif S$.

We then have the following problem:

$
  &arg min_(dif S) ip(G, dif S) space "s.t." norm(dif S)=gamma \
  ( <==>& gamma dot arg min_(dif S) ip(G, dif S) "s.t." norm(dif S) =1)\
$<P>

== Linear approach, $dif S =dif S_"linear"$
We have
$
  ip(G, dif S)&= ip(G, X(W_Q dif W_K^top + dif W_Q W_K^top ) X^top) \
  &= ip(X^top G X, W_Q dif W_K^top +dif W_Q W_K^top) space ", let" T=X^top G X \
  &= ip(T, W_Q dif W_K^top) + ip(T, dif W_Q W_K^top) \
  &= ip(dif W_Q, T W_K)+ip(dif W_K, T^top W_Q) \
$

Linear in $dif W_Q, dif W_K$.

The problem now is
$
  arg min_(dif W_Q, dif W_K) ip(dif W_Q, T W_K)+ip(dif W_K, T^top W_Q) space "s.t." norm(X(W_Q dif W_K^top +dif W_Q W_K^top ) X^top)=gamma
$

#emoji.fire The following is false, to change..

Hence the "raw directions" of steepest descent to minimize the scalar products are
$
  Delta W_Q^(( 0)) =-T W_K\
  Delta W_K^(( 0)) = -T^top W_Q
$

We define the linear operator
$
  cal(A) (Delta W_Q^((0) ) ,Delta W_K^((0) ) ) :=X(W_Q Delta W_K^top + Delta W_Q W_K^top ) X^top
$

and
$
  dif S^((0) ) :=cal(A) (Delta W_Q^((0) ) ,Delta W_K^((0) ) ), space rho:= norm(dif S^(( 0)))_F
$
We make the hypothesis that $rho !=0$, as we just have to skip the update if it is $0$.

We define
$
  alpha:=gamma / rho
$
and
$
  Delta W_Q:= alpha Delta W_Q^((0) ) , space Delta W_K^((0) ) :=alpha Delta W_K^((0) )
$

We then have
$
  norm(cal(A) (Delta W_Q ,Delta W_K ))_F=alpha rho=gamma
$
so the pair $Delta W_Q, Delta W_K$ have the best minimizing direction for the problem @eq:P, while respecting the norm constraint.

We the have the closed form expressions
$
  rho &= X(W_Q (-T^top W_Q)^top - T W_K W_K^top ) X^top \
  &= -X(W_Q W_Q^top T + T W_K W_K^top ) X^top \
  Delta W_Q^star &= - gamma / rho T W_K \
  Delta W_K^star &= - gamma / rho T^top W_Q\
$


== Quadratic approach, $dif S =dif S_"full"$
We can define
$
  dif S (x) &= X(W_Q + x dif W_Q)(W_K +x dif W_K)^top X^top -X W_Q W_K^top X^top \
  &= X(x W_Q dif W_K^top +x dif W_Q W_K^top +x^2 dif W_Q dif W_K^top ) X^top
$

Using first order approximation, should we study: (with $G = gradient_(S) cal(L)(S)$)
$
  cal(L) (S+dif S(gamma) ) =cal(L)(S) + ip(G, dif S(gamma)) + o(norm((dif S(gamma) )) )
$
// We will find $gamma$ later, with a line search.

== Problem A
We have $X in RR^( d_s times d_e)$, $G = gradient_(S) cal(L)(S) in RR^(d_s times d_s)$, $W_Q "and" W_K in RR^(d_e times d_k)$, $d_e > d_k$, $gamma in (0,oo)$.

The problem is:
$
  arg min_(gamma, dif S(gamma) ) ip(G, dif S(gamma))
$
We have
$
  ip(G, dif S(gamma)) &= ip(G, X(gamma W_Q dif W_K^top + gamma dif W_Q W_K^top + gamma^2 dif W_Q dif W_K^top ) X^top) \
  &= gamma ip(X^top G X, W_Q dif W_K^top + dif W_Q W_K^top + gamma dif W_K dif W_K^top) \
$

Let $T=X^top G X$, $R(gamma) = W_Q dif W_K^top + dif W_Q W_K^top + gamma dif W_K dif W_K^top$

We have $"rank"(R(gamma) ) = d_k < "rank"(T)$

The problem now is:
$
  arg min_(gamma, R(gamma) ) gamma ip(T, R(gamma)) "s.t." "rank"(T)>"rank"(R(gamma) ) \
$

== Problem B
We have a self-attention block. $X$ is the input, $d_s$ the sequence length, $d_e$ the embedding size, $d_k$ the key/query size.

We have $X in RR^( d_s times d_e)$, $G = gradient_(S) cal(L)(S) in RR^(d_s times d_s)$, $W_Q "and" W_K in RR^(d_e times d_k)$, $d_e > d_k$.

The idea is start with a low $d_k$ hence low expressivity, and "grow new neurons", by increasing $d_k$ by $p$.

Let $Z'= (W_Q + gamma dif W_Q)(W_K+ gamma dif W_K)^top$, $"rank"(Z')=d_k$.

We want to find the augmented matrix $Z$, such that $"rank"(Z)=d_k+p$. We basically concatenate $p$ new columns to the matrices $(W_Q + gamma dif W_Q)$ and $(W_K + gamma dif W_K)$, to augment their expressive possibility.

#emoji.fire Question: What would be the best expression for $Z$, to respect the previously introduced "step" $gamma$?
$
  Z= [W_Q + gamma dif W_Q bar gamma W_Q^("new") ] [W_K + gamma dif W_K bar gamma W_K^("new") ]^top ?
$


#emoji.fire Would augmenting $Z'$ into $Z$ cause problems with the first order approximation?

The problem is:
$
  arg min_(Z ) ip(X^top G X, Z - W_Q W_K^top)
$



== Study of $dif S$
=== Brouillon: Searching for bounds
We can find an upper bound for the quadratic term, we have, according to @ineqFrobSpecProduct:
$
  norm(dif S_"quad")_F := norm(X dif W_Q dif W_K^top X^top)_F <= norm(X)_2 norm(dif W_Q dif W_K^top)_F\
$
we have
$
  norm(dif W_Q dif W_K^top)_F <= norm(dif W_Q)_F norm(dif W_K^top)_2= norm(dif W_Q)_F norm(dif W_K)_2<= norm(dif W_Q)_F norm(dif W_K)_F\
$
hence,
$
  norm(dif S_"quad")_F <= norm(X)_2 norm(dif W_Q)_F norm(dif W_K)_F
$

We also have an upper bound for the linear term, (useless?)
$
  norm(dif S_"linear")_F:= norm(X(W_Q dif W_K^top + dif W_Q W_K^top ) X^top)_F& <= norm(X W_Q dif W_K^top X^top)_F + norm(X dif W_Q W_K^top X^top)_F\
  &<= norm(X)_2 norm(W_Q)_F norm(dif W_K)_F + norm(X)_2 norm(dif W_Q)_F norm(W_K)_F\
  &<= norm(X)_2 (norm(W_Q)_F norm(dif W_K)_F+ norm(dif W_Q)_F norm(W_K)_F)
$

And a lower bound,
$
  norm(dif S_"linear") &>= abs(norm(X W_Q dif W_K^top X^top)_F - norm(X dif W_Q W_K^top X^top)_F)
$

Hence
$
  norm(dif S_"quad") / norm(dif S_"linear") <= (norm(X)_2 norm(dif W_Q)_F norm(dif W_K)_F) / (abs((norm(X W_Q dif W_K^top X^top)_F - norm(X dif W_Q W_K^top X^top)_F )) )
$

=== Brouillon: Other bound attempt
Applying @ineqFrobSpecProduct, we have
$
  (sigma_min (X)^2 ) / (sigma_max (X)^2 ) norm(dif W_Q dif W_K^top)_F / norm(W_Q dif W_K^top + dif W_Q W_K^top)_F <= norm(dif S_"quad") / norm(dif S_"linear") <= (sigma_max (X)^2 ) / (sigma_min (X)^2 ) norm(dif W_Q dif W_K^top)_F / norm(W_Q dif W_K^top + dif W_Q W_K^top)_F
$
Hence if $sigma_min (X)$ and $sigma_max (X)$ are close,
$
  norm(dif S_"quad") / norm(dif S_"linear") approx norm(dif W_Q dif W_K^top)_F / norm(W_Q dif W_K^top + dif W_Q W_K^top)_F
$



=== Direct form
We also have the direct form
$
  norm(dif S_"quad") / norm(dif S_"linear") = (norm(X dif W_Q dif W_K^top X^top)_F) / (norm(X(W_Q dif W_K^top + dif W_Q W_K^top ) X^top)_F)
$



We can consider two different approaches, either picking $dif S_"full"$ or $dif S_"linear"$.

#emoji.fire How and when to choose $dif S_"full"$ or $dif S_"linear"$?

#show: appendices
=
#lemma[
  Let $M in RR^(m times n)$, $sigma_1 >=...>= sigma_(min(m, n) )$ its singular in decreasing order, and $M=U Sigma V^top$ its SVD decomposition.
  $
    norm(M)^2_F&=tr(M M^top)
    = tr(U Sigma V^top V Sigma^top U^top)
    = tr(U Sigma Sigma^top U^top)
    = tr(U^top U Sigma Sigma^top) \
    &= tr(Sigma Sigma^top)
    = norm(Sigma)^2_F =sum_(i)^(min(m, n) ) sigma_i^2 \
    &>= sigma_1 = norm(M)^2_2
  $
  Hence $norm(M)_F>= norm(M)_2$.
]<ineqFrobSpec>

#lemma[
  We know (bound on the Rayleigh quotient) that for any symmetric positive semidefinite matrix $M$ and any vector $x$,
  $
    x^top M x <= lambda_max (M) norm(x)^2.
  $
  Let $A in RR^(m times p)$, $B in RR^(p times n)$, $(a_1,...,a_m)$ the row vectors of $A$.

  Then
  $
    norm(A B)^2_F&=tr(A (B B^top ) A^top) \
    &= sum_(i=1)^(m) a_(i) (B B^top ) a_t^top \
    &<= sum_(i=1)^(m) lambda_max (B B^top ) norm(a_(i))^2 = lambda_max (B B^top ) tr(A A^top) =norm(B)^2_2 norm(A)^2_F
  $
  Hence $norm(A B)_F <= norm(B)_2 norm(A)_F$.

  We can prove the same way that $norm(A B)_F >= sigma_min (B) norm(A)_F$.

  The same reasoning can be applied to prove $norm(A B)_F <= norm(A)_2 norm(B)_F$.

  Moreover, let $C in RR^(n times o)$, we have
  $
    norm(A B C)_F = norm((A B) C)_F <=norm(A B)_F norm(C)_2 <= norm(A)_2 norm(B)_F norm(C)_2.
  $
]<ineqFrobSpecProduct>

