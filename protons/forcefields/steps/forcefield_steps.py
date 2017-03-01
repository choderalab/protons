from behave import given, when, then
import os
import subprocess
import shlex


@given(u'I use the current directory as working directory')
def step_use_curdir_as_working_directory(context):
    """
    Uses the current directory as working directory
    """
    context.workdir = os.path.abspath(".")
    assert os.path.exists(context.workdir), "The current path does not exist."
    assert os.path.isdir(context.workdir), "The working directory does not exist."


@given(u'the directory "{directory}" exists')
def step_the_directory_exists(context, directory):
    """
    Test if a directory exists
    """
    path = directory
    if not os.path.isabs(directory):
        path = os.path.join(context.workdir, os.path.normpath(directory))
    assert(os.path.isdir(path)), "The directory does not exist."


@given(u'a file named "{filename}" exists')
def step_a_file_named_filename_exists(context, filename):
    """Test if a file exists"""
    filename_ = os.path.join(context.workdir, filename)
    assert os.path.exists(filename_), "The file {} does not exist.".format(filename_)
    assert os.path.isfile(filename_), "Target {} is not a file.".format(filename_)


@given(u'a file named "{filename}" exists in the directory "{directory}"')
def step_a_file_named_filename_exists_in_the_directory_directory(context, filename, directory):
    """Test if a file exists"""
    filename_ = os.path.join(context.workdir, directory, filename)
    assert os.path.exists(filename_), "The file {} does not exist.".format(filename_)
    assert os.path.isfile(filename_), "Target {} is not a file.".format(filename_)


@when(u'I run "{command}"')
def step_i_run_command(context, command):
    """Run a command"""
    command_ = shlex.split(command)
    context.output = subprocess.check_output(command_,stderr=subprocess.DEVNULL)


@when(u'I successfully run "{command}"')
def step_i_successfully_run_command(context, command):
    """Check that the command is ran successfully"""
    try:
        step_i_run_command(context, command)
    except subprocess.CalledProcessError as error:
        context.return_code = error.returncode
        assert(context.return_code == 0), "The command {} did not run successfully.\n {}".format(error.cmd, error.output)


@then(u'a file named "{filename}" should exist')
def step_a_file_named_filename_should_exist(context,filename):
    """Test if a file exists"""
    step_a_file_named_filename_exists(context,filename)


@then(u'a temporary file named "{filename}" should exist')
def step_a_temporary_file_named_filename_should_exist(context,filename):
    """Test if a file exists"""
    step_a_file_named_filename_should_exist(context, filename)
    context.tmpfiles.append(os.path.join(context.workdir, filename))


@then(u'a file named "{filename}" should exist in the directory "{directory}"')
def step_a_file_named_filename_should_exist_in_the_directory_directory(context,filename, directory):
    """Test if a file exists"""
    step_a_file_named_filename_exists_in_the_directory_directory(context,filename, directory)


@then(u'a temporary file named "{filename}" should exist in the directory "{directory}"')
def step_a_temporary_file_named_filename_should_exist_in_the_directory_directory(context,filename, directory):
    """Test if a file exists"""
    step_a_file_named_filename_should_exist_in_the_directory_directory(context, filename, directory)
    context.tmpfiles.append(os.path.join(context.workdir, directory, filename))