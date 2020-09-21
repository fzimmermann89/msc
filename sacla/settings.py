Tais2020={
    'alias': {
        'pulse_energy_hutch_joule': 'bl_3_eh_5_photodiode_direct_bm_1_pulse_energy_in_joule',
        'pulse_energy_beam_joule':'bl_3_oh_2_bm_1_pulse_energy_in_joule',
        'sampleThX' : 'xfel_bl_3_st_5_motor_user_1/position',
        'sampleThY' : 'xfel_bl_3_st_5_motor_user_5/position',
        'sampleThZUpper' : 'xfel_bl_3_st_5_motor_user_4/position',
        'sampleThZLower' : 'xfel_bl_3_st_5_motor_user_2/position',
        'sampleX' : 'xfel_bl_3_st_5_motor_user_6/position',
        'sampleZ' : 'xfel_bl_3_st_5_motor_user_3/position',
        'profX' : 'xfel_bl_3_st_5_motor_user_19/position',
        'profZ' : 'xfel_bl_3_st_5_motor_user_18/position',
        'profY' : 'xfel_bl_3_st_5_motor_facility_10/position',		
        'octalX' : 'xfel_bl_3_st_5_motor_user_22/position',
        'octalY' : 'xfel_bl_3_st_5_motor_user_23/position',
        'octalZ' : 'xfel_bl_3_st_5_motor_user_24/position',	
        'dualX' : 'xfel_bl_3_st_5_motor_user_25/position',
        'dualZ' : 'xfel_bl_3_st_5_motor_user_26/position',		
        'singleX' : 'xfel_bl_3_st_5_motor_user_13/position',
        'singleZ' : 'xfel_bl_3_st_5_motor_user_14/position',
        'attenuator_eh_2_Al_thickness_in_meter':'bl_3_eh_2_attenuator_Al_thickness_in_meter',
        'attenuator_eh_4_Al_thickness_in_meter':'bl_3_eh_4_attenuator_Al_thickness_in_meter',
        'attenuator_eh_5_Si_uncalibrated':'xfel_bl_3_st_5_motor_facility_29/position',
        'attenuator_oh_2_Si_thickness_in_meter': 'bl_3_oh_2_attenuator_Si_thickness_in_meter',
        'photonEnergy':'bl_3_oh_2_photon_energy_in_eV',
#         'detector_spectrum':'detector_2d_1',
#         'detector_dual1': 'detector_2d_2',
#         'detector_dual2': 'detector_2d_3',
#         'detector_ocatal1': 'detector_2d_4',
#         'detector_octal2': 'detector_2d_5',
#         'detector_octal3': 'detector_2d_6',
#         'detector_octal4': 'detector_2d_7',
#         'detector_octal5': 'detector_2d_8',
#         'detector_octal6': 'detector_2d_9',
#         'detector_octal7': 'detector_2d_10',
#         'detector_octal8': 'detector_2d_11',
#         'detector_single': 'detector_2d_12',
#         'detector_dual_assembled':'detector_2d_assembled_1',
#         'detector_octal_assembled':'detector_2d_assembled_2'
     },
    'hide': [
      '.*camera.*',
      '.*bl_3_.*',
    ],
    'calibration':{
      'sampleThY':0.00067,
      'sampleThZUpper':0.002,
      'sampleX':0.5e-6,
      'sampleZ':0.25e-6,
      'sampleThZLower':0.002,
      'sampleThX':0.000765,	
        'octalX' : 1e-6,
        'octalY' : 1e-6,
        'octalZ' : 0.25e-6,	
        'dualX' : 5e-6,
        'dualZ' : 0.1e-6,	
      'profX':.25e-6,
      'profZ':0.005e-6, #might be 0.05 instead
      'profY':0.25e-6,
      }
 }






Tais2019={
    'alias': {
        'pulse_energy_hutch_joule': 'bl_3_eh_5_photodiode_direct_bm_1_pulse_energy_in_joule',
        'pulse_energy_beam_joule':'bl_3_oh_2_bm_1_pulse_energy_in_joule',
        'pd_volts':'xfel_bl_3_st_5_pd_user_5_fitting_peak_voltage',
        'pd_gain':'xfel_bl_3_st_5_pd_user_5_fitting_peak_gain',
        'sampleThY':'xfel_bl_3_st_5_motor_user_5/position',
        'sampleThZUpper':'xfel_bl_3_st_5_motor_user_3/position',
        'sampleX':'xfel_bl_3_st_5_motor_user_1/position', 
        'sampleZ':'xfel_bl_3_st_5_motor_user_2/position',
        'sampleThZLower':'xfel_bl_3_st_5_motor_user_4/position',
        'sampleThX':'xfel_bl_3_st_5_motor_user_15/position',
        'profX2':'xfel_bl_3_st_5_motor_facility_14/position',
        'profZ1':'xfel_bl_3_st_5_motor_user_18/position',
        'profY':'xfel_bl_3_st_5_motor_facility_10/position',
        'DualX':'xfel_bl_3_st_5_motor_user_11/position', 
        'DualZ':'xfel_bl_3_st_5_motor_user_12/position', 
        'DualThx':'xfel_bl_3_st_5_motor_user_13/position', 
        'BeamstopX':'xfel_bl_3_st_5_motor_user_14/position', 
        'BeamstopZ':'xfel_bl_3_st_5_motor_user_16/position', 
        'SingleX':'xfel_bl_3_st_5_motor_user_6/position', 
        'SingleY':'xfel_bl_3_st_5_motor_user_7/position', 
        'SingleZ':'xfel_bl_3_st_5_motor_user_8/position', 
        'SingleThy':'xfel_bl_3_st_5_motor_user_10/position', 
        'SingleThz':'xfel_bl_3_st_5_motor_user_9/position',
        'HccdX': 'xfel_bl_3_st_5_motor_facility_67/position', 
        'HccdZ': 'xfel_bl_3_st_5_motor_facility_68/position',
        'PDX': 'xfel_bl_3_st_5_motor_facility_69/position',
        'xfel_pulse_selector_status':'bl_3_eh_1_xfel_pulse_selector_status',
        'attenuator_eh_2_Al_thickness_in_meter':'bl_3_eh_2_attenuator_Al_thickness_in_meter',
        'attenuator_eh_4_Al_thickness_in_meter':'bl_3_eh_4_attenuator_Al_thickness_in_meter',
        'attenuator_eh_5_Si_thickness_in_meter':'bl_3_eh_5_attenuator_Si_thickness_in_meter',
        'attenuator_oh_2_Si_thickness_in_meter': 'bl_3_oh_2_attenuator_Si_thickness_in_meter'
     },
    'hide': [
      '.*camera.*',
      '.*bl_3_.*'
    ],
    'calibration':{
      'sampleThY':0.00067,
      'sampleThZUpper':0.002,
      'sampleX':0.5e-6,
      'sampleZ':0.25e-6,
      'sampleThZLower':0.002,
      'sampleThX':0.000765,
      #'profX2':'xfel_bl_3_st_5_motor_facility_14/position',
      #'profZ1':'xfel_bl_3_st_5_motor_user_18/position',
      #'profY':'xfel_bl_3_st_5_motor_facility_10/position',
      'DualX':5e-6, 
      'DualZ':0.1e-6,
      'DualThx':0.000706,
      'BeamstopX':1e-6,
      'BeamstopZ':0.05e-6,
      'SingleX':1e-6,
      'SingleY':1e-6,
      'SingleZ':0.1e-6,
      'SingleThy':0.0011,
      'SingleThz':0.002,
      'HccdX': 1e-6,
      'HccdZ': 0.1e-6,
      'PDX': 0.5e-6
    }
 }



