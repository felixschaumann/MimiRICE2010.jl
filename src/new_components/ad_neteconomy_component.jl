@defcomp ad_neteconomy begin
    regions = Index()

    YNET = Variable(index=[time, regions]) # Output net of damages equation (trillions 2005 USD per year)
    Y = Variable(index=[time, regions]) # Gross world product net of abatement and damages (trillions 2005 USD per year)
    I = Variable(index=[time, regions]) # Investment (trillions 2005 USD per year)
    C = Variable(index=[time, regions]) # Consumption (trillions 2005 US dollars per year)
    CPC = Variable(index=[time, regions]) # Per capita consumption (thousands 2005 USD per year)
    ADAPTCOST = Parameter(index=[time, regions]) # Cost of adaptation (trillions 2005 USD per year)
    TOTAL_COST = Variable(index=[time, regions]) # Total cost of abatement and adaptation as fraction of YGROSS
    
    YGROSS = Parameter(index=[time, regions]) # Gross world product GROSS of abatement and damages (trillions 2005 USD per year)
    DAMFRAC = Parameter(index=[time, regions]) # Damages as fraction of gross output
    DAMAGES = Parameter(index=[time, regions]) # Damages (trillions 2005 USD per year)
    ABATECOST = Parameter(index=[time, regions]) # Cost of emissions reductions  (trillions 2005 USD per year)
    AB_AD_SYN = Parameter() # Abatement and adaptation synergy factor (fraction of adaptation cost that is also counted as abatement cost and vice versa)
    S = Parameter(index=[time, regions]) # Gross savings rate as fraction of gross world product
    l = Parameter(index=[time, regions]) # Level of population and labor
    # T_AD = Parameter(index=[time, regions]) # Global temperature that a region is adapted to (degrees C from 1900)
    
    # Cost constraint parameters
    COST_CONSTRAINT_PENALTY = Variable(index=[time, regions]) # Penalty for exceeding cost cap
    COST_CAP_FRACTION = Parameter() # Maximum allowed combined cost as fraction of YGROSS (e.g., 0.02 = 2% of GDP)
    CONSTRAINT_PENALTY_STRENGTH = Parameter(default=0.0) # Strength of constraint penalty (0 = no constraint)

    function run_timestep(p, v, d, t)

        #Define function for YNET
        for r in d.regions
            if is_first(t)
                v.YNET[t,r] = p.YGROSS[t,r]/(1+p.DAMFRAC[t,r])
            else
                v.YNET[t,r] = p.YGROSS[t,r] - p.DAMAGES[t,r]
            end
        end

        #Define function for Y
        for r in d.regions
            v.Y[t,r] = v.YNET[t,r] - p.ABATECOST[t,r] - p.ADAPTCOST[t,r] + p.AB_AD_SYN * min(p.ADAPTCOST[t,r], p.ABATECOST[t,r])
        end

        #Calculate cost constraint penalty
        for r in d.regions
            v.TOTAL_COST[t,r] = (p.ABATECOST[t,r] + p.ADAPTCOST[t,r]) / p.YGROSS[t,r]
            cost_cap = p.COST_CAP_FRACTION
            excess_cost = v.TOTAL_COST[t,r] - cost_cap

            if excess_cost > 0 && p.CONSTRAINT_PENALTY_STRENGTH > 0
                # Quadratic penalty that grows with constraint violation
                v.COST_CONSTRAINT_PENALTY[t,r] = p.CONSTRAINT_PENALTY_STRENGTH * (1000 * excess_cost)^4
            else
                v.COST_CONSTRAINT_PENALTY[t,r] = 0.0
            end
        end

        #Define function for I
        for r in d.regions
            v.I[t,r] = p.S[t,r] * v.Y[t,r]
        end

        #Define function for C
        for r in d.regions
            if t.t != 60
                v.C[t,r] = v.Y[t,r] - v.I[t,r]# - p.ABATECOST[t,r] - p.ADAPTCOST[t,r] + p.AB_AD_SYN * min(p.ADAPTCOST[t,r], p.ABATECOST[t,r])
            else
                v.C[t,r] = v.C[t-1, r]
            end
        end

        #Define function for CPC
        for r in d.regions
            v.CPC[t,r] = 1000 * v.C[t,r] / p.l[t,r]
        end
    end
end