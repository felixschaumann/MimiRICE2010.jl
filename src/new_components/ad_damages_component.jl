@defcomp ad_damages begin
    regions = Index()

    DAMFRAC = Variable(index=[time, regions]) # Damages as % of GDP
    DAMAGES = Variable(index=[time, regions]) # Damages (trillions 2005 USD per year)

    HAS_STOCK_AD = Parameter{Bool}(default=true)
    T_AD_FLOW = Parameter(index=[time, regions])
    I_AD = Parameter(index=[time, regions])
    T_AD_STOCK = Variable(index=[time, regions])
    T_AD_COMBINED = Variable(index=[time, regions])
    OPT_T_SHIFT = Parameter() # Optimal temperature shift parameter for damages calculation
    dk = Parameter(index=[regions])

    T_AD = Parameter(index=[time, regions]) # Global temperature that a region is adapted to (degrees C from 1900)
    ADAPTFRAC = Parameter(index=[time, regions]) # AD-DICE-like fraction of global warming that a given region is adapted to (between 0 and 1 - 1 being full adaptation)
    ADAPTCOST = Variable(index=[time, regions]) # Cost of adaptation (trillions 2005 USD per year)

    TATM = Parameter(index=[time]) # Increase temperature of atmosphere (degrees C from 1900)
    YGROSS = Parameter(index=[time, regions]) # Gross world product GROSS of abatement and damages (trillions 2005 USD per year)
    SLRDAMAGES = Parameter(index=[time, regions])
    a1 = Parameter(index=[regions]) # Damage intercept
    a2 = Parameter(index=[regions]) # Damage quadratic term
    a3 = Parameter(index=[regions]) # Damage exponent

    function run_timestep(p, v, d, t)

        #Define function for DAMFRAC
        for r in d.regions

            if p.HAS_STOCK_AD
                if is_first(t)
                    v.T_AD_STOCK[t,r] = 0.
                else
                    v.T_AD_STOCK[t,r] = v.T_AD_STOCK[t-1,r] + (p.I_AD[t-1,r]/10)
                end
                v.T_AD_COMBINED[t,r] = (0.5 * v.T_AD_STOCK[t,r]^0.5 + 0.5 * p.T_AD_FLOW[t,r]^0.5)^2

                v.DAMFRAC[t,r] = ((p.a1[r] * (p.TATM[t] - v.T_AD_COMBINED[t, r])) +
                                  (p.a2[r] * ((p.TATM[t] - p.OPT_T_SHIFT*v.T_AD_COMBINED[t, r])^p.a3[r] - ((1-p.OPT_T_SHIFT) * v.T_AD_COMBINED[t, r])^p.a3[r]))) / 100 +
                                  p.SLRDAMAGES[t,r] / 100
                # v.DAMFRAC[t,r] = ((p.a1[r] * (p.TATM[t]- v.T_AD_COMBINED[t, r])) +
                #                   (p.a2[r] * (p.TATM[t]^p.a3[r] - v.T_AD_COMBINED[t, r]^p.a3[r]))) / 100 +
                #                   p.SLRDAMAGES[t,r] / 100
            else
                v.DAMFRAC[t,r] = ((p.a1[r] * (p.TATM[t]- p.T_AD[t, r])) +
                                  (p.a2[r] * ((p.TATM[t] - p.OPT_T_SHIFT*p.T_AD[t, r])^p.a3[r] - ((1-p.OPT_T_SHIFT) * p.T_AD[t, r])^p.a3[r]))) / 100 +
                                  p.SLRDAMAGES[t,r] / 100
                # v.DAMFRAC[t,r] = ((p.a1[r] * (p.TATM[t]- p.T_AD[t, r])) +
                #                   (p.a2[r] * (p.TATM[t]^p.a3[r] - p.T_AD[t, r]^p.a3[r]))) / 100 +
                #                   p.SLRDAMAGES[t,r] / 100
                # implement instead old AD-DICE damages
                # v.DAMFRAC[t,r] = (1 - p.ADAPTFRAC[t, r]) * (((p.a1[r] * p.TATM[t]) + (p.a2[r] * (p.TATM[t]^p.a3[r]))) / 100) + (p.SLRDAMAGES[t,r] / 100) 
            end
            
            if v.DAMFRAC[t, r] < 0
                v.DAMFRAC[t, r] = 0.0
            end

        end

        #Define function for ADAPTCOST
        for r in d.regions

            if p.HAS_STOCK_AD
                v.ADAPTCOST[t,r] = (0.003 *  p.T_AD_FLOW[t, r]^3.6 + 0.003 * p.I_AD[t, r]^3.6) * p.YGROSS[t,r] # combined at equal levels: 0.006 = 0.6% of GDP to adapt to 1°C of global warming
            else 
                v.ADAPTCOST[t,r] = 0.006 * p.T_AD[t, r]^3.6 * p.YGROSS[t,r] # usually 0.006 = 0.6% of GDP to adapt to 1°C of global warming
                # v.ADAPTCOST[t,r] = 0.115 * p.ADAPTFRAC[t, r]^3.6 * p.YGROSS[t,r] # taken from de Bruin et al. (2009)
            end
        end
        

        #Define function for DAMAGES
        for r in d.regions
            if is_first(t)
                v.DAMAGES[t,r] = p.YGROSS[t,r] * (1 - 1 / (1+v.DAMFRAC[t,r]))
            else
                v.DAMAGES[t,r] = (p.YGROSS[t,r] * v.DAMFRAC[t,r]) / (1. + v.DAMFRAC[t,r]^10)
            end
        end
    end
end