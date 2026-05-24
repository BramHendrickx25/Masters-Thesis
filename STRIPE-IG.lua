simion.workbench_program()

local runs = 1 -- Amount of runs
local ions_info = true -- If true ions are terminated at IG_testplane_4 and if false first hit principle stands (from the moment the first ion hits the IG_testplane_4 all others are killed)
local neutron_mode = false -- If true the positions and potentials felt by the ion are recorded for each time step, used to record potentials using neutral probe particle
local record_trajectory = true -- If true, record the positions of the ions for each timestep

-- z position of the test planes
-- STRIPE exit (right before cross) at 24 mm
-- STRIPE exit (right after cross) at 29 mm
-- IG center at 139.205 mm

local STRIPE_testplane = 29
local IG_testplane_1 = 49
local IG_testplane_2 = 79
local IG_testplane_3 = 109
local IG_testplane_4 = 139.205

-- RF parameters
adjustable RF_STRIPE_pp = 100  -- RF STRIPE voltage peak to peak
adjustable RF_IG_pp = 15       -- RF ion guide voltage peak to peak
adjustable nu = 1.2            -- RF frequency in MHz of IG and STRIPE
adjustable nu2 = 10            -- RF frequency in MHz of BICEPS
local RF_phase = 0             -- RF phase of initial potential
local RF_phase_offset = 0      -- RF phase shift of IG compared to STRIPE in periods

-- DC potentials for STRIPE electrode segments (trapping)
local DC_STRIPE_1_trap = 9.4
local DC_STRIPE_2_trap = 08.8
local DC_STRIPE_3_trap = 08.8
local DC_STRIPE_4_trap = 11.8

-- DC potentials for STRIPE electrode segments (extraction)
local DC_STRIPE_1_extract = 9.4
local DC_STRIPE_2_extract = 08.8
local DC_STRIPE_3_extract = 08.8
local DC_STRIPE_4_extract = 5.8

-- Time to switch from trapping to extraction
local switch_time = 20000    -- µs

-- Time to switch RF off in STRIPE
local switch_RF_time = 20000 -- µs

-- RC rise time
local rise_time = 0.100   -- µs

-- Flag to indicate if the switch has occurred
local switched = false

-- DC potential for STRIPE cross
local DC_STRIPE_Cross = 0

-- DC potentials for IG electrode segment
local DC_IG = 0

-- Simulation parameters
local simulation_start = 10000 -- Time after which the trajectory recording starts if in record_trajectory mode
local simulation_length = 250

-- Ion and transition parameters
local mass = 88
local mass_conversion = 1.66053907 * 10^(-27)
local charge = 1
local c = 299792.458
local TWO_PI = 2 * math.pi
local STABLE_TOF_THRESHOLD = 100

-- Define arrays to store data for each plane
local x1, y1, z1, vx1, vy1, vz1, xprime1, yprime1, tof1 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local x2, y2, z2, vx2, vy2, vz2, xprime2, yprime2, tof2 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local x3, y3, z3, vx3, vy3, vz3, xprime3, yprime3, tof3 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local x4, y4, z4, vx4, vy4, vz4, xprime4, yprime4, tof4 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local x5, y5, z5, vx5, vy5, vz5, xprime5, yprime5, tof5 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local ions_transported1, ions_transported2, ions_transported3, ions_transported4, ions_transported5 = 0, 0, 0, 0, 0
local step = 0
local ions_potentials = {}
local ions_x, ions_y, ions_z, ions_px, ions_py, ions_pz, ions_tof = {}, {}, {}, {}, {}, {}, {}

-- Custom defined functions
function average(array)
        local result = 0
        for _,a in ipairs(array) do result = result + a end
        if #array ~= 0 then result = result / #array end
        return result
end

function compute_x_emittance(x, xprime, vx, vy, vz)
    -- Compute average of an array
    -- returns 0 if array contains no elements
    function average(array)
        local result = 0
        for _,a in ipairs(array) do result = result + a end
        if #array ~=0 then result = result / #array end
        return result
    end

    -- compute various averages for emittance.
    local x_ave = average(x)
    local xprime_ave = average(xprime)
    local t = {}; for n = 1,#x do t[n] = (x[n] - x_ave)^2 end
    local dx2_ave = average(t)
    local t = {}; for n = 1,#x do t[n] = (xprime[n] - xprime_ave)^2 end
    local dxprime2_ave = average(t)
    local t = {}; for n = 1,#x do t[n] = (x[n]-x_ave)*(xprime[n]-xprime_ave) end
    local dx_dxprime_ave = average(t)

    -- Compute emittance from averages, in correct units.
    local m = dx2_ave * dxprime2_ave - dx_dxprime_ave^2
    if m < 0 then m = 0 end        -- safety on numerical roundoff
    local x_emit = sqrt(m) * 1000  -- (mm * mrad)

    -- Compute average speed for normalized emittance.
    local t = {}; for n = 1,#x do t[n] = sqrt(vx[n]^2 + vy[n]^2 + vz[n]^2) end
    local v_avg = average(t)
    -- Compute normalized emittance from averages
    local c = 299792.458                -- speed of light (mm/usec)
    local beta = v_avg / c              -- relativistic beta
    local gamma = 1 / sqrt(1 - beta^2)  -- relativistic gamma
    local norm_x_emit = beta * gamma * x_emit

    return x_emit, norm_x_emit
end

function compute_y_emittance(y, yprime, vx, vy, vz)
    -- Compute average of all numbers in given array.
    -- Returns 0 if array contains zero elements.
    function average(array)
        local result = 0
        for _,a in ipairs(array) do result = result + a end
        if #array ~= 0 then result = result / #array end
        return result
    end

    -- Compute various averages for emittance.
    local y_ave = average(y)
    local yprime_ave = average(yprime)
    local t = {}; for n = 1,#y do t[n] = (y[n] - y_ave)^2 end
    local dy2_ave = average(t)
    local t = {}; for n = 1,#y do t[n] = (yprime[n] - yprime_ave)^2 end
    local dyprime2_ave = average(t)
    local t = {}; for n = 1,#y do t[n] = (y[n]-y_ave)*(yprime[n]-yprime_ave) end
    local dy_dyprime_ave = average(t)

    -- Compute emittance from averages, in correct units.
    local m = dy2_ave * dyprime2_ave - dy_dyprime_ave^2
    if m < 0 then m = 0 end        -- safety on numerical roundoff
    local y_emit = sqrt(m) * 1000  -- (mm * mrad)

    -- Compute average speed for normalized emittance.
    local t = {}; for n = 1,#y do t[n] = sqrt(vx[n]^2 + vy[n]^2 + vz[n]^2) end
    local v_avg = average(t)

    -- compute normalized emittance from averages
    local c = 299792.458                -- speed of light (mm/usec)
    local beta = v_avg / c              -- relativistic beta
    local gamma = 1 / sqrt(1 - beta^2)  -- relativistic gamma
    local norm_y_emit = beta * gamma * y_emit

    return y_emit, norm_y_emit
end

function compute_average_kinetic_energy(vx,vy,vz,mass)
    -- Compute the average of an array
    function average(array)
        local result = 0
        for _,a in ipairs(array) do result = result + a end
        if #array ~= 0 then result = result / #array end
        return result
    end
    
    -- Compute the root mean square of the velocities
    local t = {}; for n =1,#vx do t[n] = math.sqrt(vx[n]^2 + vy[n]^2 + vz[n]^2) end
    local v_avg = average(t)

    -- Compute the kinetic energy
    local e_kin_avg = speed_to_ke(v_avg,mass) -- relativistic kinetic energy (eV)

    return e_kin_avg
end

function compute_kinetic_energy(vx,vy,vz,mass)
    -- Compute rms of all velocities
    local rms_v = {}; for n =1,#vx do rms_v[n] = math.sqrt(vx[n]^2 + vy[n]^2 + vz[n]^2) end

    -- Compute kinetic energies
    local e_kin = {}; for n =1,#vx do e_kin[n] = speed_to_ke(rms_v[n],mass) end -- relativistic kinetic energy (eV)

    return e_kin
end

-- This function should loop over multiple fly instances.
function segment.flym()

    -- Initialize IG-BICEPS.fly2 file if needed
    if not ions_info then
        local filename = "IG-BICEPS.fly2"
        do
            local file = io.open(filename, "w")
            file:write("particles {\n")
            file:close()
        end
	else

    	-- Initialize files for STRIPE_testplane
    	do
        	local file = io.open("Stability_data_STRIPE.csv", "w")
        	file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        	file:close()
    	end
    	do
        	local file = io.open("rawdata_STRIPE.csv", "w")
        	file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs)\n")
        	file:close()
    	end

    	-- Initialize files for IG_testplane_1
    	do
        	local file = io.open("Stability_data_IG_1.csv", "w")
        	file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        	file:close()
    	end
    	do
        	local file = io.open("rawdata_IG_1.csv", "w")
        	file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs)\n")
        	file:close()
    	end

    	-- Initialize files for IG_testplane_2
    	do
        	local file = io.open("Stability_data_IG_2.csv", "w")
        	file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        	file:close()
		end
    	do
        	local file = io.open("rawdata_IG_2.csv", "w")
        	file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs)\n")
        	file:close()
    	end

    	-- Initialize files for IG_testplane_3
    	do
        	local file = io.open("Stability_data_IG_3.csv", "w")
        	file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        	file:close()
    	end
    	do
        	local file = io.open("rawdata_IG_3.csv", "w")
        	file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs)\n")
        	file:close()
	    end

    	-- Initialize files for IG_testplane_4
    	do
        	local file = io.open("Stability_data_IG_4.csv", "w")
        	file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        	file:close()
    	end
    	do
        	local file = io.open("rawdata_IG_4.csv", "w")
        	file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs)\n")
        	file:close()
    	end

    	-- Initialize files for potentials cross-section
    	do
        	local file = io.open("potentials.csv", "w")
        	file:write("axial position(mm),potential(V)\n")
        	file:close()
    	end

    	-- Initialize files for trajectory data
    	do
        	local file = io.open("trajectories.csv", "w")
        	file:write("x_positions(mm),y_positions(mm),z_positions(mm),x_velocity(mm/usec),y_velocity(mm/usec),z_velocity(mm/usec),ToF(usec)\n")
        	file:close()
    	end
	end

    RF_phase = 0 -- Initial phase of the STRIPE and IG RF potentials
    sim_rerun_flym = 1 -- Speeds up the simulation
    sim_trajectory_image_control = 0
    for i = 1, runs do
        number_run = i
        run()
        RF_phase = RF_phase + 1/12 -- Change the initial RF phase between different runs
    end
end

-- Function to calculate the potentials during transition with risetime
function calculate_transition_potential(trap_potential, extract_potential, current_time, switch_time, rise_time)
    if current_time < switch_time then
        return trap_potential
    else
        local delta_time = current_time - switch_time
        local transition_fraction = 1 - math.exp(-delta_time / rise_time)
        return trap_potential + (extract_potential - trap_potential) * transition_fraction
    end
end

-- This function is called exactly once at the start of each run in the segment.flym() function
function segment.initialize_run()
    sim_trajectory_quality = 25 -- Can be changed according to the required quality
    sim_grouped = 1 -- Required for ion-ion interaction

    -- counters
    ions_counter = 0
    ions_stable = 0
    step = 0

    -- reset termination and termination conditions
    terminate = 0
    first_hit = false
    switched = false
    print("---")
    print('running initialisation')
    print("---")

    -- reset RF potential
    RF_STRIPE_pp = 100

    -- Define arrays to store data for each plane
    x1, y1, z1, vx1, vy1, vz1, xprime1, yprime1, tof1 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    x2, y2, z2, vx2, vy2, vz2, xprime2, yprime2, tof2 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    x3, y3, z3, vx3, vy3, vz3, xprime3, yprime3, tof3 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    x4, y4, z4, vx4, vy4, vz4, xprime4, yprime4, tof4 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    x5, y5, z5, vx5, vy5, vz5, xprime5, yprime5, tof5 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    ions_transported1, ions_transported2, ions_transported3, ions_transported4, ions_transported5 = 0, 0, 0, 0, 0

	-- Define arrays for the neutron mode and trajectory recoding mode
    ions_potentials = {}
    ions_x, ions_y, ions_z, ions_px, ions_py, ions_pz, ions_tof = {}, {}, {}, {}, {}, {}, {}
end

-- Adjusts the potentials on all the electrodes during the flym
function segment.fast_adjust()

    -- turn off RF when some time is reached
    if Ion_Time_of_Flight not false and Ion_Time_of_Flight >= switch_RF_time then
        RF_STRIPE_pp = 0
    end

    -- STRIPE RF
    RF_STRIPE = RF_STRIPE_pp/2 * math.cos(TWO_PI * (nu * Ion_Time_of_Flight + RF_phase))

    -- IG RF
    RF_IG = RF_IG_pp/2 * math.cos(TWO_PI * (nu * Ion_Time_of_Flight + RF_phase + RF_phase_offset))

    -- Calculate transition potentials for STRIPE electrodes
    local current_time = Ion_Time_of_Flight
    DC_STRIPE_1 = calculate_transition_potential(DC_STRIPE_1_trap, DC_STRIPE_1_extract, current_time, switch_time, rise_time)
    DC_STRIPE_2 = calculate_transition_potential(DC_STRIPE_2_trap, DC_STRIPE_2_extract, current_time, switch_time, rise_time)
    DC_STRIPE_3 = calculate_transition_potential(DC_STRIPE_3_trap, DC_STRIPE_3_extract, current_time, switch_time, rise_time)
    DC_STRIPE_4 = calculate_transition_potential(DC_STRIPE_4_trap, DC_STRIPE_4_extract, current_time, switch_time, rise_time)

    -- STRIPE ejection electrodes
    adj_elect01 = DC_STRIPE_1 - RF_STRIPE
    adj_elect02 = DC_STRIPE_1 + RF_STRIPE
    adj_elect03 = DC_STRIPE_2 - RF_STRIPE
    adj_elect04 = DC_STRIPE_2 + RF_STRIPE
    adj_elect05 = DC_STRIPE_3 - RF_STRIPE
    adj_elect06 = DC_STRIPE_3 + RF_STRIPE
    adj_elect07 = DC_STRIPE_4 - RF_STRIPE
    adj_elect08 = DC_STRIPE_4 + RF_STRIPE

    -- STRIPE ejection aperture
    adj_elect09 = DC_STRIPE_Cross

    -- IG electrodes
    adj_elect10 = DC_IG
    adj_elect11 = DC_IG
end

-- Import the test plane library
local TP = simion.import 'testplanelib.lua'

-- Define first four planes, STRIPE_testplane, IG_testplane_1, IG_testplane_2, IG_testplane_3. These are only used to record ion data
if ions_info then
    test_plane1 = TP(0, 0, STRIPE_testplane, 0, 0, -1,
        function()
            mark()
            print('In test plane 1: ion number = ' .. ion_number)
            ions_transported1 = ions_transported1 + 1
            x1[ions_transported1] = ion_px_mm
            y1[ions_transported1] = ion_py_mm
            z1[ions_transported1] = ion_pz_mm
            vx1[ions_transported1] = ion_vx_mm
            vy1[ions_transported1] = ion_vy_mm
            vz1[ions_transported1] = ion_vz_mm
            xprime1[ions_transported1] = ion_vx_mm / ion_vz_mm
            yprime1[ions_transported1] = ion_vy_mm / ion_vz_mm
            tof1[ions_transported1] = Ion_Time_of_Flight
        end
    )
    test_plane2 = TP(0, 0, IG_testplane_1, 0, 0, -1,
        function()
            mark()
            print('In IG_testplane_1: ion number = ' .. ion_number)
            ions_transported2 = ions_transported2 + 1
            x2[ions_transported2] = ion_px_mm
            y2[ions_transported2] = ion_py_mm
            z2[ions_transported2] = ion_pz_mm
            vx2[ions_transported2] = ion_vx_mm
            vy2[ions_transported2] = ion_vy_mm
            vz2[ions_transported2] = ion_vz_mm
            xprime2[ions_transported2] = ion_vx_mm / ion_vz_mm
            yprime2[ions_transported2] = ion_vy_mm / ion_vz_mm
            tof2[ions_transported2] = Ion_Time_of_Flight
        end
    )
    test_plane3 = TP(0, 0, IG_testplane_2, 0, 0, -1,
        function()
            mark()
            print('In IG_testplane_2: ion number = ' .. ion_number)
            ions_transported3 = ions_transported3 + 1
            x3[ions_transported3] = ion_px_mm
            y3[ions_transported3] = ion_py_mm
            z3[ions_transported3] = ion_pz_mm
            vx3[ions_transported3] = ion_vx_mm
            vy3[ions_transported3] = ion_vy_mm
            vz3[ions_transported3] = ion_vz_mm
            xprime3[ions_transported3] = ion_vx_mm / ion_vz_mm
            yprime3[ions_transported3] = ion_vy_mm / ion_vz_mm
            tof3[ions_transported3] = Ion_Time_of_Flight
        end
    )
    test_plane4 = TP(0, 0, IG_testplane_3, 0, 0, -1,
        function()
            mark()
            print('In IG_testplane_3: ion number = ' .. ion_number)
            ions_transported4 = ions_transported4 + 1
            x4[ions_transported4] = ion_px_mm
            y4[ions_transported4] = ion_py_mm
            z4[ions_transported4] = ion_pz_mm
            vx4[ions_transported4] = ion_vx_mm
            vy4[ions_transported4] = ion_vy_mm
            vz4[ions_transported4] = ion_vz_mm
            xprime4[ions_transported4] = ion_vx_mm / ion_vz_mm
            yprime4[ions_transported4] = ion_vy_mm / ion_vz_mm
            tof4[ions_transported4] = Ion_Time_of_Flight
        end
    )
end

-- Fifth plane IG_testplane_4 can either function as the first hit plane or a termination plane as described beside the ions_info boolean
local test
if not ions_info then
    test_plane5 = TP(0, 0, IG_testplane_4, 0, 0, -1,
        function()
            if not first_hit then
                mark()
                print('In IG_testplane_4: ion number = ' .. ion_number)
                first_hit = true
            end
        end
    )
else
    test_plane5 = TP(0, 0, IG_testplane_4, 0, 0, -1,
        function()
            mark()
            print('In IG_testplane_4: ion number = ' .. ion_number)
            ions_transported5 = ions_transported5 + 1
            ion_splat = 1
            x5[ions_transported5] = ion_px_mm
            y5[ions_transported5] = ion_py_mm
            z5[ions_transported5] = ion_pz_mm
            vx5[ions_transported5] = ion_vx_mm
            vy5[ions_transported5] = ion_vy_mm
            vz5[ions_transported5] = ion_vz_mm
            xprime5[ions_transported5] = ion_vx_mm / ion_vz_mm
            yprime5[ions_transported5] = ion_vy_mm / ion_vz_mm
            tof5[ions_transported5] = Ion_Time_of_Flight
        end
    )
end

-- Merge the test planes' other_actions with my existing logic
local original_other_actions = segment.other_actions

-- function to apply some viscous damping
local viscous_damping = 1
local damping = 0
function segment.accel_adjust()
    if damping == 0 then return end
    local nt = damping * ion_time_step
    local factor = (1 - math.exp(-nt)) / nt
    ion_ax_mm = (ion_ax_mm - ion_vx_mm * viscous_damping) * factor
    ion_ay_mm = (ion_ay_mm - ion_vy_mm * viscous_damping) * factor
    ion_az_mm = (ion_az_mm - ion_vz_mm * viscous_damping) * factor
end

segment.other_actions = function()

    if neutron_mode then
        if step % 1000 == 0 then  -- Only append every 1000 steps
            ions_potentials[step/1000] = ion_volts
            ions_z[step/1000] = ion_pz_mm
        end
        step = step + 1
    end

    if ions_info then
        test_plane1.other_actions()
        test_plane2.other_actions()
        test_plane3.other_actions()
        test_plane4.other_actions()
    end
    test_plane5.other_actions()

    if first_hit then
        ion_splat = 1
        ions_transported5 = ions_transported5 + 1
        x5[ions_transported5] = ion_px_mm
        y5[ions_transported5] = ion_py_mm
        z5[ions_transported5] = ion_pz_mm
        vx5[ions_transported5] = ion_vx_mm
        vy5[ions_transported5] = ion_vy_mm
        vz5[ions_transported5] = ion_vz_mm
        xprime5[ions_transported5] = ion_vx_mm / ion_vz_mm
        yprime5[ions_transported5] = ion_vy_mm / ion_vz_mm
        tof5[ions_transported5] = Ion_Time_of_Flight
    end

    if record_trajectory and Ion_Time_of_Flight > simulation_start then
        ions_x[step] = ion_px_mm
        ions_y[step] = ion_py_mm
        ions_z[step] = ion_pz_mm
	ions_px[step] = ion_vx_mm
	ions_py[step] = ion_vy_mm
	ions_pz[step] = ion_vz_mm
	ions_tof[step] = Ion_Time_of_Flight
        step = step + 1
    end

    if Ion_Time_of_Flight > simulation_length + simulation_start then
        ion_splat = 1
        ions_stable = ions_stable + 1
        z1[ions_stable] = ion_pz_mm
        print("x-positie: " .. string.format(ion_px_mm).. " mm")
        print("y-positie: " .. string.format(ion_py_mm).. " mm")
        print("z-positie: " .. string.format(ion_pz_mm).. " mm")
    end

    if ion_splat ~= 0 then
        ions_counter = ions_counter + 1
    end
end

-- Adjust time step
function segment.tstep_adjust()
    if ions_info then
        test_plane1.tstep_adjust()
        test_plane2.tstep_adjust()
        test_plane3.tstep_adjust()
        test_plane4.tstep_adjust()
    end
    test_plane5.tstep_adjust()
    ion_time_step = math.min(ion_time_step, 0.1/nu2)

    if record_trajectory then
	ion_time_step = 0.1/nu2
    end

end

function segment.terminate_run()
    if terminate == 0 then
        -- Calculate transport percentage for each plane
        local transport_percent1 = (ions_transported1 / ions_counter) * 100
        print("Transported through STRIPE_testplane: " .. string.format("%.2f", transport_percent1) .. "%")
        print("---")

		-- STRIPE_testplane
        if transport_percent1 ~= 0 then
            local x_emit1, norm_x_emit1 = compute_x_emittance(x1, xprime1, vx1, vy1, vz1)
            local y_emit1, norm_y_emit1 = compute_y_emittance(y1, yprime1, vx1, vy1, vz1)
            local total_emit1 = math.sqrt(x_emit1 * y_emit1)
            local total_norm_emit1 = norm_x_emit1 * norm_y_emit1
            print("Beam Emittance (STRIPE_testplane) = " .. total_emit1 .. " mm * mrad (Normalized = " .. total_norm_emit1 .. ")")

            local e_kin_avg1 = compute_average_kinetic_energy(vx1, vy1, vz1, mass)
            print("Average kinetic energy (STRIPE_testplane) = " .. e_kin_avg1 .. " eV")

            local r1 = {}
            for n = 1, #x1 do
                r1[n] = math.sqrt((x1[n] - 16)^2 + (y1[n] - 16)^2)
            end
            local r_avg1 = average(r1)
            print(string.format("%.4f", r_avg1) .. " mm (STRIPE_testplane)")

            local tof_spread1 = tof1[#tof1] - tof1[1]
            print(string.format("%.4f", tof_spread1) .. " usec (STRIPE_testplane)")

            local file1 = io.open("Stability_data_STRIPE.csv", "a")
            if not file1 then
                print("Failed to open file for writing (STRIPE_testplane).")
            else
                io.output(file1)
                io.write(
                    string.format("%.2f", transport_percent1) .. "," ..
                    tostring(total_norm_emit1) .. "," ..
                    tostring(e_kin_avg1) .. "," ..
                    tostring(r_avg1) .. "," ..
                    tostring(tof_spread1) .. "\n"
                )
                file1:close()
            end

            local raw_file1 = io.open("rawdata_STRIPE.csv", "a")
            if not raw_file1 then
                print("Failed to open raw data file for writing (STRIPE_testplane).")
            else
                for i = 1, #x1 do
                    local line = string.format(
                        "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        x1[i], y1[i], z1[i], vx1[i], vy1[i], vz1[i],
                        speed_to_ke(vx1[i], mass),
                        speed_to_ke(vy1[i], mass),
                        speed_to_ke(vz1[i], mass),
                        tof1[i]
                    )
                    raw_file1:write(line)
                end
                raw_file1:close()
            end
        end

        -- IG_testplane_1
        local transport_percent2 = (ions_transported2 / ions_counter) * 100
        print("Transported through IG_testplane_1: " .. string.format("%.2f", transport_percent2) .. "%")
        if transport_percent2 ~= 0 then
            local x_emit2, norm_x_emit2 = compute_x_emittance(x2, xprime2, vx2, vy2, vz2)
            local y_emit2, norm_y_emit2 = compute_y_emittance(y2, yprime2, vx2, vy2, vz2)
            local total_emit2 = math.sqrt(x_emit2 * y_emit2)
            local total_norm_emit2 = norm_x_emit2 * norm_y_emit2
            print("Beam Emittance (IG_testplane_1) = " .. total_emit2 .. " mm * mrad (Normalized = " .. total_norm_emit2 .. ")")

            local e_kin_avg2 = compute_average_kinetic_energy(vx2, vy2, vz2, mass)
            print("Average kinetic energy (IG_testplane_1) = " .. e_kin_avg2 .. " eV")

            local r2 = {}
            for n = 1, #x2 do
                r2[n] = math.sqrt((x2[n] - 16)^2 + (y2[n] - 16)^2)
            end
            local r_avg2 = average(r2)
            print(string.format("%.4f", r_avg2) .. " mm (IG_testplane_1)")

            local tof_spread2 = tof2[#tof2] - tof2[1]
            print(string.format("%.4f", tof_spread2) .. " usec (IG_testplane_1)")

            local file2 = io.open("Stability_data_IG_1.csv", "a")
            if not file2 then
                print("Failed to open file for writing (IG_testplane_1).")
            else
                io.output(file2)
                io.write(
                    string.format("%.2f", transport_percent2) .. "," ..
                    tostring(total_norm_emit2) .. "," ..
                    tostring(e_kin_avg2) .. "," ..
                    tostring(r_avg2) .. "," ..
                    tostring(tof_spread2) .. "\n"
                )
                file2:close()
            end

            local raw_file2 = io.open("rawdata_IG_1.csv", "a")
            if not raw_file2 then
                print("Failed to open raw data file for writing (IG_testplane_1).")
            else
                for i = 1, #x2 do
                    local line = string.format(
                        "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        x2[i], y2[i], z2[i], vx2[i], vy2[i], vz2[i],
                        speed_to_ke(vx2[i], mass),
                        speed_to_ke(vy2[i], mass),
                        speed_to_ke(vz2[i], mass),
                        tof2[i]
                    )
                    raw_file2:write(line)
                end
                raw_file2:close()
            end
        end

        -- IG_testplane_2
        local transport_percent3 = (ions_transported3 / ions_counter) * 100
        print("Transported through IG_testplane_2: " .. string.format("%.2f", transport_percent3) .. "%")
        if transport_percent3 ~= 0 then
            local x_emit3, norm_x_emit3 = compute_x_emittance(x3, xprime3, vx3, vy3, vz3)
            local y_emit3, norm_y_emit3 = compute_y_emittance(y3, yprime3, vx3, vy3, vz3)
            local total_emit3 = math.sqrt(x_emit3 * y_emit3)
            local total_norm_emit3 = norm_x_emit3 * norm_y_emit3
            print("Beam Emittance (IG_testplane_2) = " .. total_emit3 .. " mm * mrad (Normalized = " .. total_norm_emit3 .. ")")

            local e_kin_avg3 = compute_average_kinetic_energy(vx3, vy3, vz3, mass)
            print("Average kinetic energy (IG_testplane_2) = " .. e_kin_avg3 .. " eV")

            local r3 = {}
            for n = 1, #x3 do
                r3[n] = math.sqrt((x3[n] - 16)^2 + (y3[n] - 16)^2)
            end
            local r_avg3 = average(r3)
            print(string.format("%.4f", r_avg3) .. " mm (IG_testplane_2)")

            local tof_spread3 = tof3[#tof3] - tof3[1]
            print(string.format("%.4f", tof_spread3) .. " usec (IG_testplane_2)")

            local file3 = io.open("Stability_data_IG_2.csv", "a")
            if not file3 then
                print("Failed to open file for writing (Plane 3).")
            else
                io.output(file3)
                io.write(
                    string.format("%.2f", transport_percent3) .. "," ..
                    tostring(total_norm_emit3) .. "," ..
                    tostring(e_kin_avg3) .. "," ..
                    tostring(r_avg3) .. "," ..
                    tostring(tof_spread3) .. "\n"
                )
                file3:close()
            end

            local raw_file3 = io.open("rawdata_IG_2.csv", "a")
            if not raw_file3 then
                print("Failed to open raw data file for writing (IG_testplane_2).")
            else
                for i = 1, #x3 do
                    local line = string.format(
                        "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        x3[i], y3[i], z3[i], vx3[i], vy3[i], vz3[i],
                        speed_to_ke(vx3[i], mass),
                        speed_to_ke(vy3[i], mass),
                        speed_to_ke(vz3[i], mass),
                        tof3[i]
                    )
                    raw_file3:write(line)
                end
                raw_file3:close()
            end
        end

        -- IG_testplane_3
        local transport_percent4 = (ions_transported4 / ions_counter) * 100
        print("Transported through IG_testplane_3: " .. string.format("%.2f", transport_percent4) .. "%")
        if transport_percent4 ~= 0 then
            local x_emit4, norm_x_emit4 = compute_x_emittance(x4, xprime4, vx4, vy4, vz4)
            local y_emit4, norm_y_emit4 = compute_y_emittance(y4, yprime4, vx4, vy4, vz4)
            local total_emit4 = math.sqrt(x_emit4 * y_emit4)
            local total_norm_emit4 = norm_x_emit4 * norm_y_emit4
            print("Beam Emittance (IG_testplane_3) = " .. total_emit4 .. " mm * mrad (Normalized = " .. total_norm_emit4 .. ")")

            local e_kin_avg4 = compute_average_kinetic_energy(vx4, vy4, vz4, mass)
            print("Average kinetic energy (IG_testplane_3) = " .. e_kin_avg4 .. " eV")

            local r4 = {}
            for n = 1, #x4 do
                r4[n] = math.sqrt((x4[n] - 16)^2 + (y4[n] - 16)^2)
            end
            local r_avg4 = average(r4)
            print(string.format("%.4f", r_avg4) .. " mm (IG_testplane_3)")

            local tof_spread4 = tof4[#tof4] - tof4[1]
            print(string.format("%.4f", tof_spread4) .. " usec (IG_testplane_3)")

            local file4 = io.open("Stability_data_IG_3.csv", "a")
            if not file4 then
                print("Failed to open file for writing (IG_testplane_3).")
            else
                io.output(file4)
                io.write(
                    string.format("%.2f", transport_percent4) .. "," ..
                    tostring(total_norm_emit4) .. "," ..
                    tostring(e_kin_avg4) .. "," ..
                    tostring(r_avg4) .. "," ..
                    tostring(tof_spread4) .. "\n"
                )
                file4:close()
            end

            local raw_file4 = io.open("rawdata_IG_3.csv", "a")
            if not raw_file4 then
                print("Failed to open raw data file for writing (IG_testplane_3).")
            else
                for i = 1, #x4 do
                    local line = string.format(
                        "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        x4[i], y4[i], z4[i], vx4[i], vy4[i], vz4[i],
                        speed_to_ke(vx4[i], mass),
                        speed_to_ke(vy4[i], mass),
                        speed_to_ke(vz4[i], mass),
                        tof4[i]
                    )
                    raw_file4:write(line)
                end
                raw_file4:close()
            end
        end

        -- IG_testplane_4
        local transport_percent5 = (ions_transported5 / ions_counter) * 100
        print("Transported through IG_testplane_4: " .. string.format("%.2f", transport_percent5) .. "%")
        if transport_percent5 ~= 0 then
            local x_emit5, norm_x_emit5 = compute_x_emittance(x5, xprime5, vx5, vy5, vz5)
            local y_emit5, norm_y_emit5 = compute_y_emittance(y5, yprime5, vx5, vy5, vz5)
            local total_emit5 = math.sqrt(x_emit5 * y_emit5)
            local total_norm_emit5 = norm_x_emit5 * norm_y_emit5
            print("Beam Emittance (Plane 5) = " .. total_emit5 .. " mm * mrad (Normalized = " .. total_norm_emit5 .. ")")

            local e_kin_avg5 = compute_average_kinetic_energy(vx5, vy5, vz5, mass)
            print("Average kinetic energy (IG_testplane_4) = " .. e_kin_avg5 .. " eV")

            local r5 = {}
            for n = 1, #x5 do
                r5[n] = math.sqrt((x5[n] - 16)^2 + (y5[n] - 16)^2)
            end
            local r_avg5 = average(r5)
            print(string.format("%.4f", r_avg5) .. " mm (IG_testplane_4)")

            local tof_spread5 = tof5[#tof5] - tof5[1]
            print(string.format("%.4f", tof_spread5) .. " usec (IG_testplane_4)")

            local file5 = io.open("Stability_data_IG_4.csv", "a")
            if not file5 then
                print("Failed to open file for writing (IG_testplane_4).")
            else
                io.output(file5)
                io.write(
                    string.format("%.2f", transport_percent5) .. "," ..
                    tostring(total_norm_emit5) .. "," ..
                    tostring(e_kin_avg5) .. "," ..
                    tostring(r_avg5) .. "," ..
                    tostring(tof_spread5) .. "\n"
                )
                file5:close()
            end

            local raw_file5 = io.open("rawdata_IG_4.csv", "a")
            if not raw_file5 then
                print("Failed to open raw data file for writing (IG_testplane_4).")
            else
                for i = 1, #x5 do
                    local line = string.format(
                        "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        x5[i], y5[i], z5[i], vx5[i], vy5[i], vz5[i],
                        speed_to_ke(vx5[i], mass),
                        speed_to_ke(vy5[i], mass),
                        speed_to_ke(vz5[i], mass),
                        tof5[i]
                    )
                    raw_file5:write(line)
                end
                raw_file5:close()
            end
        end

        if not ions_info then
    		local file = io.open("IG-BICEPS.fly2", "a")
    		local ke5 = compute_kinetic_energy(vx5, vy5, vz5, mass)
    		file:write("  coordinates = 0,\n")

    		-- Calculate the RF period and the offset
    		local rf_period = 1 / 1.2  -- µs
    		local target_offset = 120   -- µs
    		-- Round to the nearest RF period
    		local rounded_offset = math.floor(target_offset / rf_period + 0.5) * rf_period

    		for i = 1, ions_transported5 do

       			-- Apply the offset for the ions in sets of 10
        		local tob_offset = (number_run - 1) * (rounded_offset + 1/12 * rf_period)
        		local adjusted_tob = tof5[i] + tob_offset

        		file:write(string.format([[
        		standard_beam {
        		n = 1,
        		tob = %f,
        		mass = %d,
        		charge = %d,
        		ke = %e,
        		cwf = 1,
        		color = 0,
        		direction = vector(%f, %f, %f),
        		position = vector(%f, %f, %f)
        		},
        		]], adjusted_tob, mass, charge, ke5[i], vx5[i], vy5[i], vz5[i], x5[i], y5[i], z5[i] - 139.205 + 10))
    		end
    		file:close()
	    end

        if neutron_mode then
            local file = io.open("potentials.csv", "a")
            if not file then
                print("Failed to open file for writing (potentials).")
            else
                -- Iterate over the lists
                for i = 1, #ions_potentials do
                    io.output(file)
                    io.write(
                        tostring(ions_z[i]) .. "," ..
                        tostring(ions_potentials[i]) .. "\n"
                    )
                end
                file:close()
            end
        end
        terminate = 1
        if number_run == runs then
            local file = io.open("IG-BICEPS.fly2", "a")
            file:write("}\n")
            file:close()
        end
    end
end

if record_trajectory then
    function segment.terminate_run()
	    local file = io.open("trajectories.csv", "a")
            if not file then
                print("Failed to open file for writing (trajectories).")
            else
                -- Iterate over the lists
                for i = 1, #ions_x do
                    io.output(file)
                    io.write(
                        tostring(ions_x[i]) .. "," ..
                        tostring(ions_y[i]) .. "," ..
                        tostring(ions_z[i]) .. "," ..
			tostring(ions_px[i]) .. "," ..
			tostring(ions_py[i]) .. "," ..
			tostring(ions_pz[i]) .. "," ..
			tostring(ions_tof[i]) .. "\n"
                    )
                end
                file:close()
            end
    end
end
