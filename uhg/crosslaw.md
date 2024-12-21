The Cross law
The Cross law is the analog of the Cosine law. In planar rational trigonometry the Cross law has the form
(Q1 − Q2 − Q3)2 = 4Q2Q3 (1 − s1) (31)
involving three quadrances and one spread, and it is hard to overstate the importance of this most powerful formula.
In the hyperbolic setting, the Cross law is more complicated, but still very fundamental.
Theorem 50 (Cross law) Suppose that a1, a2 and a3 are distinct points with quadrances q1 ≡ q (a2, a3), q2 ≡ q (a1, a3) and q3 ≡ q (a1, a2), and spread S1 ≡ S (a1a2, a1a3). Then
(q2q3S1 − q1 − q2 − q3 + 2)2 = 4 (1 − q1) (1 − q2) (1 − q3) . (32)
26
Proof. Suppose that a1 ≡ [x1 : y1 : z1], a2 ≡ [x2 : y2 : z2] and a3 ≡ [x3 : y3 : z3]. The assumption that all three
quadrances are defined implies that the three points are non-null. Square both sides of the polynomial identity
− (x1y2z3 − x1y3z2 + x2y3z1 − x3y2z1 + x3y1z2 − x2y1z3)2
+
􀀀
x2
1 + y2
1 − z2
1
 
(x2x3 + y2y3 − z2z3)2 +
􀀀
x2
2 + y2
2 − z2
2
 
(x1x3 + y1y3 − z1z3)2
+
􀀀
x2
3 + y2
3 − z2
3
 
(x1x2 + y1y2 − z1z2)2
−
􀀀
x2
1 + y2
1 − z2
1
  􀀀
x2
2 + y2
2 − z2
2
  􀀀
x2
3 + y2
3 − z2
3
 
(33)
= 2 (x2x3 + y2y3 − z2z3) (x1x3 + y1y3 − z1z3) (x1x2 + y1y2 − z1z2)
and divide by 􀀀
x2
1 + y2
1 − z2
1
 2 􀀀
x2
2 + y2
2 − z2
2
 2 􀀀
x2
3 + y2
3 − z2
3
 2
to deduce that if A ≡ A(a1, a2, a3) then
(A + (1 − q1) + (1 − q2) + (1 − q3) − 1)2 = 4 (1 − q1) (1 − q2) (1 − q3) .
Rewrite this as
(A − q1 − q2 − q3 + 2)2 = 4 (1 − q1) (1 − q2) (1 − q3) (34)
and use the Quadrea theorem to replace A by q2q3S1.
The Cross law gives a quadratic equation for the spreads of a triangle given the quadrances. So the three
quadrances of a triangle do not quite determine its spreads. As a quadratic equation in A, (34) can be rewritten
using the Triple spread function as
A2 − 2 (q1 + q2 + q3 − 2)A = S (q1, q2, q3) .
Motivated by the Cross law, we define the Cross function
C (A, q1, q2, q3) ≡ (A − q1 − q2 − q3 + 2)2 − 4 (1 − q1) (1 − q2) (1 − q3) . (35)
Example 15 Suppose that a triangle a1a2a3 has equal quadrances q1 = q2 = q3 ≡ −3. Then
C (A,−3,−3,−3) = (A + 11)2 − 256 = 0
has solutions A = −27 and A = 5, and from the Quadrea theorem we deduce that
S1 = S2 = S3 = −3 or S1 = S2 = S3 =
5
9
.
Two triangles a1a2a3 that have these quadrances and spreads can be found over the respective fields Q
􀀀√2,√3
 
and
Q
􀀀√2,√3,√5
 
, with
a1 ≡
 √2 : 0 : 1
 
a2 ≡
 
−1 : √3 : √2
 
a3 ≡
 
−1 : −√3 : √2
 
and
a1 ≡
 √2 : 0 : √5
 
a2 ≡
 
−1 : √3 : √10
 
a3 ≡
 
−1 : −√3 : √10
 
. ⋄
It is an instructive exercise to verify that both of the classical hyperbolic Cosine laws
cosh d1 = cosh d2 cosh d3 − sinh d2 sinh d3 cos  1 (36)
and
cosh d1 =
cos  2 cos  3 + cos  1
sin  2 sin  3
(37)
relating lengths d1, d2, d3 and angles  1,  2,  3 in a classical hyperbolic triangle can be manipulated using (7) and
(8) to obtain the Cross law.
Theorem 51 (Cross dual law) Suppose that L1,L2 and L3 are distinct lines with spreads S1 ≡ S (L2,L3), S2 ≡ S (L1,L3) and S3 ≡ S (L1,L2), and quadrance q1 ≡ q (L1L2,L1L3). Then
(S2S3q1 − S1 − S2 − S3 + 2)2 = 4 (1 − S1) (1 − S2) (1 − S3) .
Proof. This is dual to the Cross law.
This can also be restated in terms of the quadreal L ≡ L(L1,L2,L3) as
(L − S1 − S2 − S3 + 2)2 = 4 (1 − S1) (1 − S2) (1 − S3) .
27
3.11 Alternate formulations
As in the Euclidean case, the most powerful of the trigonometric laws is the Cross law
(q2q3S1 − q1 − q2 − q3 + 2)2 = 4 (1 − q1) (1 − q2) (1 − q3) . (38)
In the special case S1 = 0, we get
(q1 + q2 + q3 − 2)2 = 4 (1 − q1) (1 − q2) (1 − q3)
which is equivalent to the Triple quad formula
(q1 + q2 + q3)2 = 2
􀀀
q2
1 + q2
2 + q2
3
 
+ 4q1q2q3.
If we rewrite (38) in the form
(q1 − q2 − q3 + q2q3S1)2 = 4q2q3 (1 − q1) (1 − S1) (39)
then in the special case S1 = 1 we recover Pythagoras’ theorem in the form
q1 = q2 + q3 − q2q3.
Also (39) may be viewed as a deformation of the planar Cross law (31).