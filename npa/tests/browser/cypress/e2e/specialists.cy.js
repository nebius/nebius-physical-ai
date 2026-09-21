/* Exercise the shipped specialist monitor's task controls and untrusted-text rendering. */
describe('Workbench specialists', () => {
  const token = 'synthetic-browser-service-credential';
  const goal = '<img src=x onerror="window.injected=true"> inspect the pipeline';
  let paused;
  const task = () => ({id: 'inspect-one', profile: 'simulation', goal, status: 'completed', paused,
    result: 'Validation passed', route: {status: 'explicit'}, calls: [], events: []});
  beforeEach(() => {
    paused = false;
    cy.intercept('GET', '/api/status', request => {
      expect(request.headers.authorization).to.equal('Bearer ' + token);
      request.reply({profiles: [{name: 'simulation', model: 'synthetic/model', description: 'Pipeline analysis', paused, worker: {seen: 1}}], tasks: [task()]});
    });
    cy.intercept('GET', '/api/tasks/inspect-one', request => request.reply(task()));
    cy.intercept('GET', '/api/tasks/inspect-one/patch', {headers: {'content-type': 'text/plain'}, body: '-broken\n+fixed\n'});
    cy.visit('/specialists/');
    cy.get('#token').type(token, {log: false});
    cy.get('#connect').submit();
    cy.get('#connection').should('have.text', 'Connected');
  });
  it('shows real receipt fields, escapes goals, and keeps credentials out of browser storage', () => {
    cy.get('#profiles').should('contain.text', 'synthetic/model');
    cy.get('#tasks').should('contain.text', goal).find('img').should('not.exist');
    cy.get('.task').click();
    cy.get('#result').should('have.text', 'Validation passed');
    cy.get('#patch').should('contain.text', '+fixed');
    cy.window().then(window => {
      expect(window.injected).to.equal(undefined);
      expect(window.localStorage.length).to.equal(0);
      expect(window.sessionStorage.length).to.equal(0);
    });
    cy.get('#token').should('have.value', '');
  });
  it('submits explicit assignments and controls profiles independently', () => {
    cy.intercept('POST', '/api/tasks', request => {
      expect(request.body.specialist).to.equal('simulation');
      expect(request.body.task_id).to.match(/^[a-f0-9-]+$/);
      request.reply(task());
    }).as('submit');
    cy.get('#specialist').select('simulation'); cy.get('#goal').type('Validate workflow');
    cy.get('#submit').submit(); cy.wait('@submit');
    cy.intercept('POST', '/api/profiles/simulation/pause', request => {
      paused = request.body.paused; request.reply({paused});
    }).as('pause');
    cy.contains('button', 'Pause specialist').click(); cy.wait('@pause');
    cy.contains('button', 'Resume specialist').should('be.visible');
  });
});
