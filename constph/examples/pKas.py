from sympy import symbols, Eq, solvers, abc
HP, HD, HE = symbols('HiP, HiD, HiE')

z = abc.z  # 10^(pH - pKa_d)
y = abc.y  # 10^(pH - pKa_e)

d = Eq(z, HD/HP)
e = Eq(y, HE/HP)
t3 = Eq(HP + HE + HD, 1.0)  # Total fractional concentration = 1.0
t2 = Eq(HP + HD, 1.0)  # Total fractional concentration = 1.0

# Two accessible deprotonated forms (e.g. HIS)
print(solvers.solve([d, e, t3], [HP, HD, HE]))
print("z = 10^(pH - pKa_d), y = 10^(pH - pKa_e)")
# One accessible deprotonated form (e.g. ASP)
print(solvers.solve([d, t2], [HP, HD]))
print("z = 10^(pH - pKa)")
