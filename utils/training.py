from utils.evaluation import calculate_loss


# TODO: refactor this function, too big, too many arguments
def train_model(
    model,
    splitted_loaders,
    criterion,
    optimizer,
    num_epochs,
    output_to=print,
):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in splitted_loaders.train:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        epoch_train_loss = calculate_loss(model, splitted_loaders.train, criterion)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = calculate_loss(model, splitted_loaders.validation, criterion)
        val_losses.append(epoch_val_loss)

        text_line = create_text_output(
            epoch, num_epochs, epoch_train_loss, epoch_val_loss
        )
        output_to(text_line, end="\r")

    output_to(" " * (len(text_line) + 1), end="\r")
    output_to(
        create_text_output(
            epoch, num_epochs, epoch_train_loss, epoch_val_loss, is_finished=True
        ),
    )
    return train_losses, val_losses


def create_text_output(
    epoch, num_epochs, epoch_train_loss, epoch_val_loss, is_finished=False
):
    if is_finished:
        return f"Finished! Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
    else:
        return f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
