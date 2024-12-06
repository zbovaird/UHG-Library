# Only changes are in the main() function's training loop and scheduler setup
def main():
    """Main training function."""
    # Load and preprocess data
    node_features, labels, label_mapping = load_and_preprocess_data()
    graph_data = create_graph_data(node_features, labels)
    
    # Model parameters
    in_channels = graph_data.x.size(1)
    hidden_channels = 128
    out_channels = len(label_mapping)
    num_layers = 2
    
    # Initialize model and optimizer
    model = UHGGraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    ).to(device)
    
    # Learning rate setup
    initial_lr = 0.01
    min_lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # More sophisticated learning rate scheduling
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        epochs=num_epochs,
        steps_per_epoch=len(range(graph_data.train_mask.sum())) // batch_size,
        pct_start=warmup_epochs/num_epochs,  # Warmup phase
        final_div_factor=initial_lr/min_lr,  # Minimum LR
        div_factor=10.0,  # Initial LR division factor
        three_phase=True  # Use three-phase learning
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 400
    best_val_acc = 0
    patience = 20
    counter = 0
    
    # Learning rate monitoring setup
    lr_history = []
    val_history = []
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        try:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            loss = train_epoch(model, graph_data, optimizer, criterion)
            
            # Evaluate
            val_acc = evaluate(model, graph_data, graph_data.val_mask)
            test_acc = evaluate(model, graph_data, graph_data.test_mask)
            
            # Store learning rate and validation accuracy
            lr_history.append(current_lr)
            val_history.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"\nNew best model saved! Validation accuracy: {val_acc:.4f}")
            else:
                counter += 1
            
            # Print progress with learning rate
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}, '
                      f'Test Accuracy: {test_acc:.4f}, Learning Rate: {current_lr:.6f}')
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered! Final learning rate: {current_lr:.6f}")
                break
                
        except RuntimeError as e:
            print(f"\nError in epoch {epoch}: {str(e)}")
            break
    
    # Print learning rate statistics
    print("\nLearning Rate Statistics:")
    print(f"Initial LR: {lr_history[0]:.6f}")
    print(f"Final LR: {lr_history[-1]:.6f}")
    print(f"Min LR: {min(lr_history):.6f}")
    print(f"Max LR: {max(lr_history):.6f}")
    
    # Final evaluation
    if os.path.exists(MODEL_SAVE_PATH):
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        final_test_acc = evaluate(model, graph_data, graph_data.test_mask)
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    main() 