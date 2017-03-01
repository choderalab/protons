# Created by rustenba at 3/1/2017
Feature: Building the forcefield files
  This feature tests that the scripts used to generate
  forcefield xml files are still functional

  Scenario: Compiling raw ffxml for constant-pH residues from amber source files
    Given I use the current directory as working directory
    And the directory "Amber_input_files" exists
    When I successfully run "python Amber_input_files/1-compile_ffxml.py"
    Then a temporary file named "raw-protons-tmp.xml" should exist in the directory "Amber_input_files"

  Scenario: Processing raw parmed output
    Given I use the current directory as working directory
    And the directory "Amber_input_files" exists
    And a file named "Amber_input_files/raw-protons.xml" exists
    When I successfully run "python Amber_input_files/2-process_raw_xml_file.py"
    Then a temporary file named "protons-tmp.xml" should exist

  Scenario: Generating hydrogen definitions
    Given I use the current directory as working directory
    And the directory "Amber_input_files" exists
    And a file named "protons.xml" exists
    When I successfully run "python Amber_input_files/3-create_hydrogen_definitions.py"
    Then a temporary file named "hydrogens-protons-tmp.xml" should exist

  Scenario: Generating bond definitions
    Given I use the current directory as working directory
    And the directory "Amber_input_files" exists
    And a file named "protons.xml" exists
    When I successfully run "python Amber_input_files/4-create_bond_definitions.py"
    Then a temporary file named "bonds-protons-tmp.xml" should exist

  Scenario: Compiling raw ffxml for amber ion parameters from amber source files
    Given I use the current directory as working directory
    And the directory "Amber_input_files/ions" exists
    When I successfully run "python Amber_input_files/ions/1-compile_ion_ffxml.py"
    Then a temporary file named "raw_ions_spce-tmp.xml" should exist in the directory "Amber_input_files/ions"
    And a temporary file named "raw_ions_tip3p-tmp.xml" should exist in the directory "Amber_input_files/ions"
    And a temporary file named "raw_ions_tip4pew-tmp.xml" should exist in the directory "Amber_input_files/ions"

  Scenario: Compiling raw ffxml for amber ion parameters from amber source files
    Given I use the current directory as working directory
    And the directory "Amber_input_files/ions" exists
    And a file named "Amber_input_files/ions/raw_ions_spce.xml" exists
    And a file named "Amber_input_files/ions/raw_ions_tip3p.xml" exists
    And a file named "Amber_input_files/ions/raw_ions_tip4pew.xml" exists
    When I successfully run "python Amber_input_files/ions/2-process_raw_ff10_ions.py"
    Then a temporary file named "ions_spce-tmp.xml" should exist
    Then a temporary file named "ions_tip3p-tmp.xml" should exist
    Then a temporary file named "ions_tip4pew-tmp.xml" should exist
