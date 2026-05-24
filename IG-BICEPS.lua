simion.workbench_program()

-- adjust how many ion crystals you want to run out of 480 crystals
local total_crystal_number = 60


-- z position of the test plane
-- 10 mm from end of IG at 109.205 mm
-- BICEPS center at 136.19 mm
local IG_testplane = 136.19
local BICEPS_testplane = 0
local BICEPS_RF_testplane = 0

-- RF parameters
adjustable RF_IG_pp = 15       -- RF ion guide voltage pole to pole
adjustable RF_BICEPS_pp = 1000  -- RF BICEPS voltage pole to pole
adjustable nu = 1.2            -- RF frequency in MHz of IG and STRIPE
adjustable nu2 = 10            -- RF frequency in MHz of BICEPS

-- DC potentials for IG electrode segments 
local DC_IG = 0

-- Aperture/holder potentials
local DC_BICEPS_Ap = 0 -- BICEPS holder

-- DC potentials for BICEPS electrode segments (trapping)
local DC_BICEPS_1_trap = 0
local DC_BICEPS_2_trap = 17.48
local DC_BICEPS_3_trap = 47.48

-- DC potentials for BICEPS electrode segments (capturing)
local DC_BICEPS_1_capture = 47.48
local DC_BICEPS_2_capture = 17.48
local DC_BICEPS_3_capture = 47.48

-- RC rise time
local rise_time = 0.100   -- µs

-- Flag to indicate if the switch has occurred
local switched = false
local switch_time = 100  -- µs (=~61 is the ion time of birth in the simulation)
local trapping_time = 40   -- µs (the amount of time the trapping potential stays on after switch)
local ions_kill_in_trap_time = 30  -- µs (amount of time ions stay in the trapping potential before being killed)
local rounded_bunch_offset = (math.floor(120 * nu + 0.5) + 1/12)/ nu

local trap_switch_time = switch_time + trapping_time
local ions_kill_switch_time = switch_time + ions_kill_in_trap_time

local ions_per_crystal = 10
local current_bunch = 0
local total_ions_number = total_crystal_number * ions_per_crystal

-- Laser and simulation parameters
Irel = 100
laser_offset = 1
local simulation_length = 200

-- Ion and transition parameters
local mass = 88
local mass_conversion = 1.66053907 * 10^(-27)
local r0 = 1.04
local c = 299792.458
local TWO_PI = 2 * math.pi
local STABLE_TOF_THRESHOLD = 200

-- Define arrays to store data for each plane
local x1, y1, z1, vx1, vy1, vz1, xprime1, yprime1, tof1 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local x2, y2, z2, vx2, vy2, vz2, xprime2, yprime2, tof2 = {}, {}, {}, {}, {}, {}, {}, {}, {}
local ions_transported1, ions_transported2, ions_transported3 = 0, 0, 0
local ions_number1 = {}
local x, y, z = {}, {}, {}
-----------------------------------------------------------------------------------------------
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

    -- Initialize files for Plane 1
    do
        local file = io.open("Stability_data_IG.csv", "w")
        file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        file:close()
    end

    do
        local file = io.open("rawdata_IG.csv", "w")
        file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs),ion-number(#)\n")
        file:close()
    end

    -- Initialize files for Plane 2
    do
        local file = io.open("Stability_data_BICEPS.csv", "w")
        file:write("Transport(%),Total_norm_emit,Av_Kinetic_Energy(eV),radial position(mm),Time-of-Flight spread(usec)\n")
        file:close()
    end

    do
        local file = io.open("rawdata_BICEPS.csv", "w")
        file:write("x-position(mm),y-position(mm),z-position(mm),x-velocity(mm/µs),y-velocity(mm/µs),z-velocity(mm/µs),x-ke(eV),y-ke(eV),z-ke(eV),time-of-flight(µs)\n")
        file:close()
    end

    sim_rerun_flym = 1  -- enable "rerun" mode, makes repeated simulations go faster as no trajectory data is retained
    sim_trajectory_image_control = 0 -- controls trajectory imaging, can speed up simulation when put off (0)
    run()
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
    sim_trajectory_quality = 25  -- Quality of the trajectory (-500,500)
    sim_grouped = 1             -- Dont Fly the ions in group

    -- counters
    ions_counter = 0
    ions_stable = 0

    -- reset termination
    terminate = 0
    print("---")
    print('running initialisation')
    print("---")

    -- Define arrays to store data for each plane
    x1, y1, z1, vx1, vy1, vz1, xprime1, yprime1, tof1 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    x2, y2, z2, vx2, vy2, vz2, xprime2, yprime2, tof2 = {}, {}, {}, {}, {}, {}, {}, {}, {}
    ions_transported1, ions_transported2, ions_transported3 = 0, 0, 0
    ions_number1 = {}
    x, y, z = {}, {}, {}
end

-- Adjusts the potentials on all the electrodes during the flym
function segment.fast_adjust()
    RF_IG = RF_IG_pp/2 * math.cos(nu * TWO_PI * Ion_Time_of_Flight)
    RF_BICEPS = RF_BICEPS_pp/2 * math.cos(nu2 * TWO_PI * Ion_Time_of_Flight)

    -- Calculate transition potentials for BICEPS electrodes
    local current_time = Ion_Time_of_Flight

    DC_BICEPS_1 = calculate_transition_potential(DC_BICEPS_1_trap, DC_BICEPS_1_capture, current_time, switch_time, rise_time)
    DC_BICEPS_2 = calculate_transition_potential(DC_BICEPS_2_trap, DC_BICEPS_2_capture, current_time, switch_time, rise_time)
    DC_BICEPS_3 = calculate_transition_potential(DC_BICEPS_3_trap, DC_BICEPS_3_capture, current_time, switch_time, rise_time)

    -- IG electrodes
    adj_elect15 = DC_IG - RF_IG
    adj_elect16 = DC_IG + RF_IG

    -- BICEPS injection aperture
    adj_elect17 = DC_BICEPS_Ap

    -- BICEPS electrodes
    adj_elect18 = DC_BICEPS_1
    adj_elect19 = RF_BICEPS
    adj_elect20 = DC_BICEPS_2
    adj_elect21 = RF_BICEPS
    adj_elect22 = DC_BICEPS_3
    adj_elect23 = RF_BICEPS
    -- BICEPS holder
    adj_elect24 = DC_BICEPS_Ap
end

-- Import the test plane library
local TP = simion.import 'testplanelib.lua'

-- First test plane: Records data as ions pass through
local test_plane1 = TP(0, 0, IG_testplane, 0, 0, -1,
  function()
    mark()
    print('In test plane 1: ion number = ' .. ion_number)
    print('ToF = ' .. Ion_Time_of_Flight)
    ions_transported1 = ions_transported1 + 1

    -- Record ion data when it hits the test plane
    x1[ions_transported1] = ion_px_mm
    y1[ions_transported1] = ion_py_mm
    z1[ions_transported1] = ion_pz_mm
    vx1[ions_transported1] = ion_vx_mm
    vy1[ions_transported1] = ion_vy_mm
    vz1[ions_transported1] = ion_vz_mm
    xprime1[ions_transported1] = ion_vx_mm / ion_vz_mm
    yprime1[ions_transported1] = ion_vy_mm / ion_vz_mm
    tof1[ions_transported1] = Ion_Time_of_Flight
    ions_number1[ions_transported1] = ion_number
  end
)

-- Second test plane: Splats ions and records data on impact
local test_plane2 = TP(0, 0, BICEPS_testplane, 0, 0, -1,
  function()
    mark()
    print('In test plane 2: ion number = ' .. ion_number)
    ion_splat = 1
    ions_transported2 = ions_transported2 + 1

    -- Record ion data when it hits the test plane
    x2[ions_transported2] = ion_px_mm
    y2[ions_transported2] = ion_py_mm
    z2[ions_transported2] = ion_pz_mm
    vx2[ions_transported2] = ion_vx_mm
    vy2[ions_transported2] = ion_vy_mm
    vz2[ions_transported2] = ion_vz_mm
    xprime2[ions_transported2] = ion_vx_mm / ion_vz_mm
    yprime2[ions_transported2] = ion_vy_mm / ion_vz_mm
    tof2[ions_transported2] = Ion_Time_of_Flight

    if ions_transported2 % 10 == 0 then
        print('ion crystal terminated')
        in_BICEPS = false
    end
  end
)

-- Third test plane: Checks if all (10) ions have passed
local test_plane3 = TP(0, 0, BICEPS_RF_testplane, 0, 0, -1,
  function()
    mark()
    ions_transported3 = ions_transported3 + 1
    if ions_transported3 % 10 == 0 then
        print('10 ions in BICEPS')
        in_BICEPS = true
    end
  end
)

-- Merge the test planes' other_actions with your existing logic
local original_other_actions = segment.other_actions

segment.other_actions = function()
    test_plane1.other_actions()
    test_plane2.other_actions()
    test_plane3.other_actions()

    -- Update switch_time when the current time exceeds trap_switch_time for the current bunch
    if Ion_Time_of_Flight > trap_switch_time then
        switch_time = switch_time + rounded_bunch_offset
	print('changing switch time to ' ..switch_time)
        trap_switch_time = switch_time + trapping_time
	print('changing trap switch time to ' ..trap_switch_time)
        ions_kill_switch_time = switch_time + ions_kill_in_trap_time
	print('changing ion kill switch time to ' ..ions_kill_switch_time)
	current_bunch = current_bunch + 1
	print('current bunch = ' .. current_bunch)
    end

    -- Splatting logic for the current target bunch
    if Ion_Time_of_Flight > ions_kill_switch_time then
	if math.floor((ion_number-1)/ions_per_crystal) == current_bunch then
        	ion_splat = 1
		print('ion killed: ' .. ion_number)
        	ions_stable = ions_stable + 1
        	x[ions_stable] = ion_px_mm
        	y[ions_stable] = ion_py_mm
        	z[ions_stable] = ion_pz_mm
	end
    end
    
    if ion_number > total_ions_number then
	ion_splat = 1
	print('splat ion number = ' .. ion_number)
    end

    if ion_splat ~= 0 and ion_number <= total_ions_number then
        ions_counter = ions_counter + 1
    end
end

-- Adjust time step
function segment.tstep_adjust()
    test_plane1.tstep_adjust()
    test_plane2.tstep_adjust()
    test_plane3.tstep_adjust()

    -- Keep time step size below some fraction of the RF period.

    ion_time_step = math.min(ion_time_step, 0.1/nu2)  -- X usec
end

function segment.terminate_run()
    if terminate == 0 then
        -- Calculate transport percentage for each plane
        local transport_percent1 = (ions_transported1 / ions_counter) * 100
        local transport_percent2 = (ions_transported2 / ions_counter) * 100

        print("Transported through Plane 1: " .. string.format("%.2f", transport_percent1) .. "%")
        print("Transported through Plane 2: " .. string.format("%.2f", transport_percent2) .. "%")

        -- Only record data if ions are transported through any plane
        if transport_percent1 ~= 0 or transport_percent2 ~= 0 then
            -- Calculate and display emittance for Plane 1
            if transport_percent1 ~= 0 then
                local x_emit1, norm_x_emit1 = compute_x_emittance(x1, xprime1, vx1, vy1, vz1)
                local y_emit1, norm_y_emit1 = compute_y_emittance(y1, yprime1, vx1, vy1, vz1)
                local total_emit1 = math.sqrt(x_emit1 * y_emit1)
                local total_norm_emit1 = norm_x_emit1 * norm_y_emit1
                print("Beam Emittance (Plane 1) = " .. total_emit1 .. " mm * mrad (Normalized = " .. total_norm_emit1 .. ")")

                -- Calculate and display average kinetic energy for Plane 1
                local e_kin_avg1 = compute_average_kinetic_energy(vx1, vy1, vz1, mass)
                print("Average kinetic energy (Plane 1) = " .. e_kin_avg1 .. " eV")

                -- Calculate radial position for Plane 1
                local r1 = {}
                for n = 1, #x1 do
                    r1[n] = math.sqrt((x1[n] - 10)^2 + (y1[n] - 10)^2)
                end
                local r_avg1 = average(r1)
                print(string.format("%.4f", r_avg1) .. " mm (Plane 1)")

                -- Display TOF spread for Plane 1
                local tof_spread1 = tof1[#tof1] - tof1[1]
                print(string.format("%.4f", tof_spread1) .. " usec (Plane 1)")
            end

            -- Calculate and display emittance for Plane 2
            if transport_percent2 ~= 0 then
                local x_emit2, norm_x_emit2 = compute_x_emittance(x2, xprime2, vx2, vy2, vz2)
                local y_emit2, norm_y_emit2 = compute_y_emittance(y2, yprime2, vx2, vy2, vz2)
                local total_emit2 = math.sqrt(x_emit2 * y_emit2)
                local total_norm_emit2 = norm_x_emit2 * norm_y_emit2
                print("Beam Emittance (Plane 2) = " .. total_emit2 .. " mm * mrad (Normalized = " .. total_norm_emit2 .. ")")

                -- Calculate and display average kinetic energy for Plane 2
                local e_kin_avg2 = compute_average_kinetic_energy(vx2, vy2, vz2, mass)
                print("Average kinetic energy (Plane 2) = " .. e_kin_avg2 .. " eV")

                -- Calculate radial position for Plane 2
                local r2 = {}
                for n = 1, #x2 do
                    r2[n] = math.sqrt((x2[n] - 10)^2 + (y2[n] - 10)^2)
                end
                local r_avg2 = average(r2)
                print(string.format("%.4f", r_avg2) .. " mm (Plane 2)")

                -- Display TOF spread for Plane 2
                local tof_spread2 = tof2[#tof2] - tof2[1]
                print(string.format("%.4f", tof_spread2) .. " usec (Plane 2)")
            end

            -- Save data for Plane 1
            if transport_percent1 ~= 0 then
                local file1 = io.open("Stability_data_IG.csv", "a")
                if not file1 then
                    print("Failed to open file for writing (Plane 1).")
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

                -- Save raw ion data for Plane 1
                local raw_file1 = io.open("rawdata_IG.csv", "a")
                if not raw_file1 then
                    print("Failed to open raw data file for writing (Plane 1).")
                else
                    for i = 1, #x1 do
                        local line = string.format(
                            "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i\n",
                            x1[i], y1[i], z1[i], vx1[i], vy1[i], vz1[i],
                            speed_to_ke(vx1[i], mass),
                            speed_to_ke(vy1[i], mass),
                            speed_to_ke(vz1[i], mass),
                            tof1[i],
			    ions_number1[i]
                        )
                        raw_file1:write(line)
                    end
                    raw_file1:close()
                end
            end

            -- Save data for Plane 2
            if transport_percent2 ~= 0 then
                local file2 = io.open("Stability_data_BICEPS.csv", "a")
                if not file2 then
                    print("Failed to open file for writing (Plane 2).")
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

                -- Save raw ion data for Plane 2
                local raw_file2 = io.open("rawdata_BICEPS.csv", "a")
                if not raw_file2 then
                    print("Failed to open raw data file for writing (Plane 2).")
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
        end
        terminate = 1
    end
end
