#! PYTHONEXE
from argparse import ArgumentParser
from chemistry.amber.readparm import AmberParm
from cpinutils import __version__
from cpinutils import residues
from cpinutils.exceptions import *
from cpinutils.residues import TitratableResidueList
from cpinutils.utilities import process_arglist
import os
from ParmedTools.ParmedActions import changeradii, change
import sys

parser = ArgumentParser(epilog='''This program will read a topology file and
                        generate a cpin file for constant pH simulations with
                        sander''', usage='%(prog)s [Options]')
parser.add_argument('-v', '--version', action='version', version='%s: %s' %
                    (parser.prog, __version__))
parser.add_argument('-d', '--debug', dest='debug', action='store_const',
                    help='Enable verbose tracebacks to debug this program',
                    const=True, default=False)
group = parser.add_argument_group('Output files')
group.add_argument('-o', '--output', dest='output', metavar='FILE',
                   help='Output file. Defaults to standard output')
group.add_argument('-op', '--output-prmtop', dest='outparm', metavar='FILE',
                   help='''For explicit solvent simulations, a custom set of
                   radii are necessary to obtain reasonable results for
                   carboxylate pKas (e.g., AS4 and GL4 residues). If specified,
                   this file will be the prmtop compatible with the reference
                   energies in the printed cpin file.''', default=None)
group = parser.add_argument_group('Required Arguments')
group.add_argument('-p', dest='prmtop', metavar='FILE', required=False,
                   help='Topology file to be used in constant pH simulation',
                   type=str, default='prmtop')
group = parser.add_argument_group('Simulation Options')
group.add_argument('-igb', dest='igb', metavar='IGB', required=False, type=int,
                   help='Generalized Born model which you intend to use to '
                   'evaluate dynamics (or protonation state swaps). Default '
                   'is 2.', default=2)
group.add_argument('-intdiel', dest='intdiel', metavar='DIEL', type=float,
                   default=1.0, help='''Internal dielectric constant to use in
                   the evaluation of the GB potential. Default %(default)s.''')
group = parser.add_argument_group('Residue Selection Options')
group.add_argument('-resnames', dest='resnames', metavar='RES', nargs='*',
                   help='Residue names to include in CPIN file', default=None)
group.add_argument('-notresnames', dest='notresnames', metavar='RES', nargs='*',
                   help='Residue names to exclude from CPIN file', default=None)
group.add_argument('-resnums', dest='resnums', metavar='NUM',
                   nargs='*', help='Residue numbers to include in CPIN file',
                   default=None)
group.add_argument('-notresnums', dest='notresnums', nargs='*', metavar='NUM',
                 help='Residue numbers to exclude from CPIN file', default=None)
group.add_argument('-minpKa', dest='minpka', metavar='pKa', type=float,
                   help='Minimum reference pKa to include in CPIN file',
                   default=-999999)
group.add_argument('-maxpKa', dest='maxpka', metavar='pKa', type=float,
                   help='Maximum reference pKa to include in CPIN file',
                   default=9999999)
group = parser.add_argument_group('System Information')
group.add_argument('-states', dest='resstates', metavar='NUM', nargs='*',
                 help='List of default states to assign to titratable residues')
group.add_argument('-system', dest='system', metavar='<system name>',
                   help='Name of system to titrate. No effect on simulation.',
                   default='Unknown')
group = parser.add_argument_group('Residue Information', '''If any options here
              are used, no CPIN file will be written. These arguments take
              precedence and are mutually exclusive with each other.''')
group.add_argument('--describe', dest='descres', metavar='RESNAME', nargs='*',
                   help='Print out the details of given residues', default=None)
group.add_argument('-l', '--list', dest='list', default=False,
                   action='store_true', help='List all titratable residues')
group.add_argument_group('Explicit Solvent Options')
#group.add_argument('--counter-ions', dest='coions', action='store_true',
#                   default=False, help='''Transform a water molecule into a
#                   chloride ion when deprotonating (or a water into a chloride
#                   while protonating) to maintain charge neutrality''')

opt = parser.parse_args()

# Set up the exception handler
replace_excepthook(opt.debug)

def print_residues(resnames):
   for resname in resnames:
      if not hasattr(residues, resname):
         print ('%s is not titratable\n' % res)
      print (str(getattr(residues, resname)) + '\n')

def list_residues():
   """ Lists all titratable residues defined in residues.py """
   from cpinutils.residues import _LineBuffer as LineBuffer
   line = LineBuffer(sys.stdout)
   line.add_words(', '.join(residues.titratable_residues).split(),
                  space_delimited=True)
   line.flush()

def main(opt):
   # Check all of the arguments
   if not os.path.exists(opt.prmtop):
      raise CpinInputError('prmtop file (%s) is missing' % opt.prmtop)
   
   # Process the arguments that take multiple args
   resstates = process_arglist(opt.resstates, int)
   resnums = process_arglist(opt.resnums, int)
   notresnums = process_arglist(opt.notresnums, int)
   resnames = process_arglist(opt.resnames, str)
   notresnames = process_arglist(opt.notresnames, str)
   minpka = opt.minpka
   maxpka = opt.maxpka
   
   if not opt.igb in (2, 5, 8):
      raise CpinInputError('-igb must be 2, 5, or 8!')
   
   if resnums is not None and notresnums is not None:
      raise CpinInputError('Cannot specify -resnums and -notresnums together')
   
   if resnames is not None and notresnames is not None:
      raise CpinInputError('Cannot specify -resnames and -notresnames together')
   
   if opt.intdiel != 1 and opt.intdiel != 2:
      raise CpinInputError('-intdiel must be either 1 or 2 currently')

   # Set the list of residue names we will be willing to titrate
   titratable_residues = []
   if notresnames is not None:
      for resname in residues.titratable_residues:
         if resname in notresnames: continue
         titratable_residues.append(resname)
   elif resnames is not None:
      for resname in resnames:
         if not resname in residues.titratable_residues:
            raise CpinInputError('%s is not a titratable residue!' % resname)
         titratable_residues.append(resname)
   else:
      titratable_residues = residues.titratable_residues[:]
   
   solvent_ions = ['WAT', 'Na+', 'Br-', 'Cl-', 'Cs+', 'F-', 'I-', 'K+', 'Li+',
                   'Mg+', 'Rb+', 'CIO', 'IB', 'MG2']
   
   # Filter titratable residues based on min and max pKa
   new_reslist = []
   for res in titratable_residues:
      if getattr(residues, res).pKa < minpka: continue
      if getattr(residues, res).pKa > maxpka: continue
      new_reslist.append(res)
   titratable_residues = new_reslist
   del new_reslist
   
   # Make sure we still have a couple residues
   if len(titratable_residues) == 0:
      raise CpinInputError('No titratable residues fit your criteria!')
   
   # Load the topology file
   parm = AmberParm(opt.prmtop)
   
   # Replace an un-set notresnums with an empty list so we get __contains__()
   if notresnums is None:
      notresnums = []
   
   # If we have a list of residue numbers, check that they're all titratable
   if resnums is not None:
      for resnum in resnums:
         if resnum > parm.ptr('nres'):
            raise CpinInputError('%s only has %d residues. (%d chosen)' % (parm,
                                 parm.ptr('nres'), resnum))
         if resnum <= 0:
            raise CpinInputError('Cannot select negative residue numbers.')
         resname = parm.parm_data['RESIDUE_LABEL'][resnum-1]
         if not resname in titratable_residues:
            raise CpinInputError('Residue number %s [%s] is not titratable' % 
                                 (resnum, resname))
   else:
      # Select every residue except those in notresnums
      resnums = []
      for resnum in range(1, parm.ptr('nres') + 1):
         if resnum in notresnums: continue
         resnums.append(resnum)
   
   solvated = False
   first_solvent = 0
   if 'WAT' in parm.parm_data['RESIDUE_LABEL']:
      solvated = True
      for i, res in enumerate(parm.parm_data['RESIDUE_LABEL']):
         if res in solvent_ions:
            first_solvent = parm.parm_data['RESIDUE_POINTER'][i]
            break
   main_reslist = TitratableResidueList(system_name=opt.system, 
                     solvated=solvated, first_solvent=first_solvent)
   for resnum in resnums:
      resname = parm.parm_data['RESIDUE_LABEL'][resnum-1]
      if not resname in titratable_residues: continue
      res = getattr(residues, resname)
      # Filter out termini (make sure the residue in the prmtop has as many
      # atoms as the titratable residue defined in residues.py)
      if resnum == parm.ptr('nres'):
         natoms = parm.ptr('natom')-parm.parm_data['RESIDUE_POINTER'][resnum-1]
      else:
         natoms = (parm.parm_data['RESIDUE_POINTER'][resnum] -
                     parm.parm_data['RESIDUE_POINTER'][resnum-1])
      if natoms != len(res.atom_list): continue
      # If we have gotten this far, add it to the list.
      main_reslist.add_residue(res, resnum,
                               parm.parm_data['RESIDUE_POINTER'][resnum-1])
   
   # Set the states if requested
   if resstates is not None:
      main_reslist.set_states(resstates)
   
   # Open the output file
   if opt.output is None:
      output = sys.stdout
   else:
      output = open(opt.output, 'w', 0)
   
   main_reslist.write_cpin(output, opt.igb, opt.intdiel)
   
   if opt.output is not None:
      output.close()
   
   if solvated:
      if opt.outparm is None:
         has_carboxylate = False
         for res in main_reslist:
            if res is residues.AS4 or res is residues.GL4:
               has_carboxylate = True
               break
         if has_carboxylate:
            print >> sys.stderr, "Warning: Carboxylate residues in explicit ",
            print >> sys.stderr, "solvent simulations require a modified"
            print >> sys.stderr, "topology file! Use the -op flag to print one."
      else:
         changeradii(parm, 'mbondi2').execute()
         change(parm, 'RADII', ':AS4,GL4@OD=,OE=', 1.3).execute()
         parm.overwrite = True
         parm.writeParm(opt.outparm)
   else:
      if opt.outparm is not None:
         print >> sys.stderr, "A new prmtop is only necessary for explicit ",
         print >> sys.stderr, "solvent CpHMD simulations."

   sys.stderr.write('CPIN generation complete!\n')

if __name__ == '__main__':
   opt = parser.parse_args()
   replace_excepthook(opt.debug)

   # List all residues
   if opt.list:
      list_residues()
      sys.exit(0)

   # Describe requested residues
   if opt.descres is not None:
      if len(opt.descres) == 0 or (len(opt.descres) == 1 and
                                   opt.descres[0].upper() == 'ALL'):
         print_residues(residues.titratable_residues)
      else:
         opt.descres = process_arglist(opt.descres, str)
         print_residues(opt.descres)
      sys.exit(0)
   
   # Go ahead and make the CPIN file.
   main(opt)
   sys.exit(0)
