# for testing equations given my claude ai

def cross_ratio(v1, v2, u1, u2):
    """
    Compute cross ratio (v₁,v₂:u₁,u₂) for projective points
    Used as foundation for quadrance calculation
    """
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    w1, m1, n1 = u1 
    w2, m2, n2 = u2
    
    return ((x1*m1 - y1*w1)*(x2*m2 - y2*w2)) / \
           ((x2*m1 - y2*w1)*(x1*m2 - y1*w2))

  def join_points(a1, a2):
    """
    Get line joining two points using hyperbolic cross product
    Returns: (l:m:n) coordinates of line
    """
    x1, y1, z1 = a1
    x2, y2, z2 = a2
    return [
        y1*z2 - y2*z1,
        z1*x2 - z2*x1,
        x2*y1 - x1*y2
    ]

def meet_lines(L1, L2):
    """
    Get point of intersection of two lines using dual of cross product
    Returns: [x:y:z] coordinates of point
    """
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    return [
        m1*n2 - m2*n1,
        n1*l2 - n2*l1,
        l2*m1 - l1*m2
    ]         

def dual_point_to_line(point):
    """Convert point [x:y:z] to its dual line (x:y:z)"""
    return point  # Same coordinates, different interpretation

def dual_line_to_point(line):
    """Convert line (l:m:n) to its dual point [l:m:n]"""
    return line  # Same coordinates, different interpretation

def batch_quadrance(points1, points2):
    """
    Compute quadrance between batches of points
    points1, points2: tensors of shape (batch_size, 3)
    Returns: tensor of shape (batch_size)
    """
    # Inner product in ambient space
    dot_product = (points1 * points2).sum(dim=-1)
    
    # Quadratic forms
    q1 = (points1 * points1).sum(dim=-1)
    q2 = (points2 * points2).sum(dim=-1)
    
    # Quadrance formula
    return 1 - dot_product**2 / (q1 * q2)

def normalized_coordinates(points):
    """
    Normalize projective coordinates to maintain numerical stability
    while preserving projective equivalence
    """
    norm = torch.norm(points, dim=-1, keepdim=True)
    return points / (norm + 1e-8)  # Small epsilon for stability

def quadrance(a1, a2):
    """
    Calculate quadrance between two points a1=[x1:y1:z1] and a2=[x2:y2:z2]
    Returns None if either point is null
    """
    x1, y1, z1 = a1
    x2, y2, z2 = a2
    
    # Check if points are null
    if x1*x1 + y1*y1 - z1*z1 == 0 or x2*x2 + y2*y2 - z2*z2 == 0:
        return None
        
    numerator = (x1*x2 + y1*y2 - z1*z2)**2
    denominator = (x1*x1 + y1*y1 - z1*z1)*(x2*x2 + y2*y2 - z2*z2)
    
    return 1 - numerator/denominator

def triple_quad_formula(q1, q2, q3):
    """
    Verifies if three quadrances satisfy the triple quad formula
    (q₁ + q₂ + q₃)² = 2(q₁² + q₂² + q₃²) + 4q₁q₂q₃
    """
    lhs = (q1 + q2 + q3)**2
    rhs = 2*(q1**2 + q2**2 + q3**2) + 4*q1*q2*q3
    return lhs == rhs

def pythagoras(q1, q2, q3):
    """
    Verifies if three quadrances satisfy Pythagoras' theorem
    q₃ = q₁ + q₂ - q₁q₂
    """
    return q3 == q1 + q2 - q1*q2

def spread(L1, L2):
    """
    Calculate spread between two lines L1=(l1:m1:n1) and L2=(l2:m2:n2)
    Returns None if either line is null
    """
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    
    # Check if lines are null
    if l1*l1 + m1*m1 - n1*n1 == 0 or l2*l2 + m2*m2 - n2*n2 == 0:
        return None
        
    numerator = (l1*l2 + m1*m2 - n1*n2)**2
    denominator = (l1*l1 + m1*m1 - n1*n1)*(l2*l2 + m2*m2 - n2*n2)
    
    return 1 - numerator/denominator

def batch_spread(lines1, lines2):
    """
    Compute spread between batches of lines
    lines1, lines2: tensors of shape (batch_size, 3)
    Returns: tensor of shape (batch_size)
    """
    # Inner product in ambient space
    dot_product = (lines1 * lines2).sum(dim=-1)
    
    # Quadratic forms
    q1 = (lines1 * lines1).sum(dim=-1)
    q2 = (lines2 * lines2).sum(dim=-1)
    
    return 1 - dot_product**2 / (q1 * q2)

def triple_spread_formula(S1, S2, S3):
    """
    Verifies if three spreads satisfy the triple spread formula
    (S₁ + S₂ + S₃)² = 2(S₁² + S₂² + S₃²) + 4S₁S₂S₃
    """
    lhs = (S1 + S2 + S3)**2
    rhs = 2*(S1**2 + S2**2 + S3**2) + 4*S1*S2*S3
    return lhs == rhs

def spread_law(S1, S2, S3, q1, q2, q3):
    """
    Verifies the spread law relation
    S₁/q₁ = S₂/q₂ = S₃/q₃
    """
    return abs(S1/q1 - S2/q2) < 1e-10 and abs(S2/q2 - S3/q3) < 1e-10

def cross_dual_law(S1, S2, S3, q1):
    """
    Verifies the cross dual law
    (S₂S₃q₁ - S₁ - S₂ - S₃ + 2)² = 4(1-S₁)(1-S₂)(1-S₃)
    """
    lhs = (S2*S3*q1 - S1 - S2 - S3 + 2)**2
    rhs = 4*(1-S1)*(1-S2)*(1-S3)
    return lhs == rhs

def spread_from_null_points(alpha1, alpha2, beta1, beta2):
    """
    Compute spread between lines determined by pairs of null points
    Important for deep learning as null points often parameterize the space
    """
    L1 = join_points(alpha1, alpha2)
    L2 = join_points(beta1, beta2)
    return spread(L1, L2)

def spread_quadrance_duality(L1, L2):
    """
    Verify that spread between lines equals quadrance between dual points
    S(L₁,L₂) = q(L₁⊥,L₂⊥)
    """
    p1 = dual_line_to_point(L1)
    p2 = dual_line_to_point(L2)
    return spread(L1, L2) == quadrance(p1, p2)

def point_lies_on_line(point, line):
    """
    Check if point [x:y:z] lies on line (l:m:n)
    lx + my - nz = 0
    """
    x, y, z = point
    l, m, n = line
    return l*x + m*y - n*z == 0

def points_perpendicular(a1, a2):
    """
    Check if points [x₁:y₁:z₁] and [x₂:y₂:z₂] are perpendicular
    x₁x₂ + y₁y₂ - z₁z₂ = 0
    """
    x1, y1, z1 = a1
    x2, y2, z2 = a2
    return x1*x2 + y1*y2 - z1*z2 == 0

def lines_perpendicular(L1, L2):
    """
    Check if lines (l₁:m₁:n₁) and (l₂:m₂:n₂) are perpendicular
    l₁l₂ + m₁m₂ - n₁n₂ = 0
    """
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    return l1*l2 + m1*m2 - n1*n2 == 0

def parametrize_line_point(L, p, r, s):
    """
    Get point on line L=(l:m:n) parametrized by p,r,s
    Returns [np-ms : ls+nr : lp+mr]
    """
    l, m, n = L
    return [
        n*p - m*s,
        l*s + n*r,
        l*p + m*r
    ]

def null_point(t, u):
    """
    Get null point parametrized by t:u
    Returns [t²-u² : 2tu : t²+u²]
    """
    return [
        t*t - u*u,
        2*t*u,
        t*t + u*u
    ]

def join_null_points(t1, u1, t2, u2):
    """
    Get line through two null points parametrized by (t₁:u₁) and (t₂:u₂)
    Returns (t₁t₂-u₁u₂ : t₁u₂+t₂u₁ : t₁t₂+u₁u₂)
    """
    return [
        t1*t2 - u1*u2,
        t1*u2 + t2*u1,
        t1*t2 + u1*u2
    ]

def are_collinear(a1, a2, a3):
    """
    Test if three points are collinear using determinant formula
    x₁y₂z₃ - x₁y₃z₂ + x₂y₃z₁ - x₃y₂z₁ + x₃y₁z₂ - x₂y₁z₃ = 0
    """
    x1, y1, z1 = a1
    x2, y2, z2 = a2
    x3, y3, z3 = a3
    
    det = (x1*y2*z3 - x1*y3*z2 + x2*y3*z1 - 
           x3*y2*z1 + x3*y1*z2 - x2*y1*z3)
    return det == 0

def are_concurrent(L1, L2, L3):
    """
    Test if three lines are concurrent using determinant
    l₁m₂n₃ - l₁m₃n₂ + l₂m₃n₁ - l₃m₂n₁ + l₃m₁n₂ - l₂m₁n₃ = 0
    """
    l1, m1, n1 = L1
    l2, m2, n2 = L2
    l3, m3, n3 = L3
    
    det = (l1*m2*n3 - l1*m3*n2 + l2*m3*n1 - 
           l3*m2*n1 + l3*m1*n2 - l2*m1*n3)
    return det == 0

def opposite_points(a1, a2):
    """
    Get opposite points o₁, o₂ of side a₁a₂
    """
    def get_o1(v1, v2):
        # (x₁x₂ + y₁y₂ - z₁z₂)v₁ - (x₁² + y₁² - z₁²)v₂
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        dot = x1*x2 + y1*y2 - z1*z2
        quad = x1*x1 + y1*y1 - z1*z1
        return [
            dot*x1 - quad*x2,
            dot*y1 - quad*y2,
            dot*z1 - quad*z2
        ]
        
    def get_o2(v1, v2):
        # (x₂² + y₂² - z₂²)v₁ - (x₁x₂ + y₁y₂ - z₁z₂)v₂
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        dot = x1*x2 + y1*y2 - z1*z2
        quad = x2*x2 + y2*y2 - z2*z2
        return [
            quad*x1 - dot*x2,
            quad*y1 - dot*y2,
            quad*z1 - dot*z2
        ]
    
    o1 = get_o1(a1, a2)
    o2 = get_o2(a1, a2)
    return o1, o2

def altitude_line(a, L):
    """
    Get altitude line through point a perpendicular to line L
    Returns aL^⊥
    """
    L_dual = dual_line_to_point(L)
    return join_points(a, L_dual)

def altitude_point(a, L):
    """
    Get altitude point on L perpendicular to point a
    Returns a^⊥L
    """
    a_dual = dual_point_to_line(a)
    return meet_lines(a_dual, L)

def parallel_line(a, L):
    """
    Get parallel line through a to line L
    Returns a(a^⊥L)
    """
    alt_point = altitude_point(a, L)
    return join_points(a, alt_point)

def parallel_point(a, L):
    """
    Get parallel point on a^⊥ to point a on L
    Returns a^⊥(aL^⊥)
    """
    alt_line = altitude_line(a, L)
    return meet_lines(dual_point_to_line(a), alt_line)

