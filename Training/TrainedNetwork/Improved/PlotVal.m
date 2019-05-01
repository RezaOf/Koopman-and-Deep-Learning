load Errors(loss).mat
semilogy(Errors.Train)
grid on
hold on
xlabel('Iteration')
ylabel('Loss')
semilogy(Errors.Val)
legend('Training Loss','Validation Loss')