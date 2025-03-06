import click

from ..services import CSPService

@click.group()
@click.pass_context
def cli(ctx, **kwargs):
    pass

# CSP Service 

@click.group(name='csp')
@click.option('--host', type=str, envvar='CSP_HOST')
@click.pass_context
def csp(ctx, **kwargs):
    service = CSPService(**kwargs)
    ctx.obj = ctx.with_resource(service)


@click.command()
@click.argument('filename', type=str)
@click.pass_context
def evaluate_model_from_file(ctx, filename=None):
    service = ctx.obj
    service.evaluate_model_from_file(filename)


cli.add_command(csp)
csp.add_command(evaluate_model_from_file)

# Analyst Service 
# TODO
# User Service 
# TODO
